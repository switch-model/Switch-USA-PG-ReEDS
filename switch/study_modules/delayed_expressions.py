"""
Patch Pyomo to perform lazy construction of constraints and expressions, with
the goal of reducing memory requirements.

This code is based on Constraint from Pyomo 6.8.0, shifted over to
IndexedConstraint (we're not that interested in non-indexed ones), with a custom
RuleDict instead of ConstraintData object, to build the constraint as needed at
solution time.

[Pyomo 6.8.1](https://github.com/Pyomo/pyomo/releases/tag/6.8.1) introduced
"templatized models" to do something similar and allow generic LaTeX printing of
constraint rules. But they require constraint formulas to be the same for every
index (no `Constraint.Skip if key not in m.SET`). See templatize_model.py for
unsuccessful code to use this feature.

This version works, but uses more memory than standard Pyomo components and runs
slower. The table below shows stats For the 4-day 134-zone model when using the
COPT AMPL solver, as reported in the Memory column of macOS Activity Monitor
(should be real + vm/compressed).
                                       delayed     delayed    delayed
                                    const+expr  const only  expr only  standard
model construction time (s)                  5          16         44        31
solver time (incl. prepping matrix) (s)    206         150                  129
post-solve time (s)                         39          21                   21
memory before calling solve() (GB)        0.87        1.77                 2.65
   - attach_data_portal=False             0.86        1.72       3.27
peak memory before starting solver (GB)   11.3         8.8                  8.9
memory while running solver (GB)           8.8         6.5                  3.7
memory used by COPT solver (GB)            4.1         4.1                  4.1

with custom NL writer:
model construction time (s)                  6                               26
solver time (incl. prepping matrix) (s)    145                              110
post-solve time (s)                         39 (why slower?)                 22
memory before calling solve() (GB)        0.87                             2.64
peak memory before starting solver (GB)   1.20                             3.07
memory while running solver (GB)          0.96                             2.77
memory used by COPT solver (GB)           3.97                             3.93

appsi_ipopt (uses appsi nl writer with ipopt ampl solver)
                                        delayed
                                  expr & constr  standard
with custom NL writer:
model construction time (s)
solver time (incl. prepping matrix) (s)
post-solve time (s)
memory before calling solve() (GB)           ?
peak memory before starting solver (GB)   5.8?      5.6
memory while running solver (GB)           5.8      5.5
memory used by solver (GB)                ~3-6

TODO: try making delayed objectives too

Maybe the higher RAM usage is due to failure to share sub-expression objects
(we get different ones with each call), and then everything is in RAM anyway
just before the problem goes to the solver (either because the ampl problem
writer gathers it in RAM before launching or because of slow garbage
collection)? But where is Pyomo storing all these objects while it runs the
solver?
"""

import sys

import pyomo.core.base.expression
from pyomo.core.base.expression import (
    IndexedExpression,
    ConstructionTimer,
    is_debug_set,
    logger,
    Initializer,
    NOTSET,
)
from pyomo.core.expr.relational_expr import EqualityExpression

import pyomo.core.base.constraint
from pyomo.core.base.constraint import IndexedConstraint, Constraint

from pyomo.core.base.indexed_component import IndexedComponent, _NotSpecified


class DelayedIndexedExpression(IndexedExpression):
    # Note: currently fails on the second pass for Switch Expressions that are
    # built via a dictionary with .pop() to remove elements the first time
    # through. Those could be converted to using special indexes instead. Or we
    # could add code to memoize them, e.g., the first time we access a
    # particular component, we ask for the same element twice. If it fails the
    # second time, we memoize that component from then on. But that could fail
    # if there's a fallback, e.g., an empty set, and the first element happens
    # to use the fallback.
    # Constraints:
    # - Satisfy_Spinning_Reserve_Up_Requirement (spinning_reserves_advanced; see makedict)
    # - Satisfy_Spinning_Reserve_Down_Requirement (spinning_reserves_advanced; see makedict)
    # + Allocate_Retrofit_Builds (retrofit)
    # Expressions:
    # + FuelCostsPerTP
    # - RetrofitCapitalCost
    # + StorageNetCharge (fixed in study_modules and switch_repo)

    def _construct_from_rule_using_setitem(self):
        # override IndexedComponent behavior to prevent calculating and saving
        # individual values (they will be returned via the RuleDict instead)
        if self._rule is None:
            return
        index = None
        rule = self._rule
        block = self.parent_block()
        try:
            if rule.constant() and self.is_indexed():
                # A constant rule could return a dict-like thing or
                # matrix that we would then want to process with
                # Initializer().  If the rule actually returned a
                # constant, then this is just a little overhead.
                self._rule = rule = Initializer(
                    rule(block, None),
                    treat_sequences_as_mappings=False,
                    arg_not_specified=NOTSET,
                )

            if rule.contains_indices():
                # The index is coming in externally; we need to validate it
                # This will crash but is never used in Switch.
                for index in rule.indices():
                    self[index] = rule(block, index)
            elif not self.index_set().isfinite():
                # If the index is not finite, then we cannot iterate
                # over it.  Since the rule doesn't provide explicit
                # indices, then there is nothing we can do (the
                # assumption is that the user will trigger specific
                # indices to be created at a later time).
                pass
            else:
                self._data = RuleDict(block, self, self.index_set(), rule)
            # next two clauses have been moved to getitem()
            # elif rule.constant():
            #     # Slight optimization: if the initializer is known to be
            #     # constant, then only call the rule once.
            #     val = rule(block, None)
            #     for index in self.index_set():
            #         self._setitem_when_not_present(index, val)
            # else:
            #     for index in self.index_set():
            #         self._setitem_when_not_present(index, rule(block, index))
        except:
            err = sys.exc_info()[1]
            logger.error(
                "Rule failed for %s '%s' with index %s:\n%s: %s"
                % (self.ctype.__name__, self.name, str(index), type(err).__name__, err)
            )
            raise

    # def __getitem__(self, idx):
    #     # code below is based on Expression.construct and
    #     # IndexedComponent._setitem_when_not_present
    #     rule = self._rule
    #     block = self.parent_block()
    #     if rule.constant():
    #         expr = rule(block, None)
    #     else:
    #         if idx in self:
    #             expr = rule(block, idx)
    #         else:
    #             raise KeyError(
    #                 f"Index '{idx}' is not valid for indexed component '{self.name}'"
    #             )
    #     obj = self._ComponentDataClass(component=self)
    #     obj.set_value(expr)

    def is_reference(self):
        # prevent treatment as a reference object because there's no stored data
        return False


class DelayedIndexedConstraint(IndexedConstraint):
    # Currently not working (some problem with set_value(Constraint.Skip) trying
    # to delete the index from the parent component, which fails if set_value()
    # is before _index = ... and does infinite recursion if it is after
    # Note: this will prevent showing of constraint upper and lower bounds
    # unless we create some fancy alternative to ConstraintData that can push
    # that info into a storage spot, but use the rule to produce the body.
    def construct(self, data=None):
        """
        Construct the expression(s) for this constraint.
        """
        # based on Constraint.construct()
        if self._constructed:
            return
        self._constructed = True

        timer = ConstructionTimer(self)
        if is_debug_set(logger):
            logger.debug("Constructing constraint %s" % (self.name))

        if self._anonymous_sets is not None:
            for _set in self._anonymous_sets:
                _set.construct()

        rule = self.rule
        try:
            # We do not (currently) accept data for constructing Constraints
            index = None
            assert data is None

            if rule is None:
                # If there is no rule, then we are immediately done.
                return

            if rule.constant() and self.is_indexed():
                raise IndexError(
                    "Constraint '%s': Cannot initialize multiple indices "
                    "of a constraint with a single expression" % (self.name,)
                )

            block = self.parent_block()
            if rule.contains_indices():
                # The index is coming in externally; we need to validate it
                for index in rule.indices():
                    self[index] = rule(block, index)
            elif not self.index_set().isfinite():
                # If the index is not finite, then we cannot iterate
                # over it.  Since the rule doesn't provide explicit
                # indices, then there is nothing we can do (the
                # assumption is that the user will trigger specific
                # indices to be created at a later time).
                pass
            else:
                # Don't construct, just setup a dictionary to pretend later
                # # Bypass the index validation and create the member directly
                # for index in self.index_set():
                #     self._setitem_when_not_present(index, rule(block, index))
                self._data = RuleDict(block, self, self.index_set(), rule)

        except Exception:
            err = sys.exc_info()[1]
            logger.error(
                "Rule failed when generating expression for "
                "Constraint %s with index %s:\n%s: %s"
                % (self.name, str(index), type(err).__name__, err)
            )
            raise
        finally:
            timer.report()

    def is_reference(self):
        # prevent treatment as a reference object because there's no stored data
        return False


class RuleDict(dict):
    def __init__(self, block, component, index_set, rule):
        # Initialize with an index_set and a rule function
        self.block = block
        self.component = component
        self.index_set = index_set
        self.rule = rule

    # Override __getitem__ to return the value based on rule(key)
    # This is based on IndexedComponent._setitem_when_not_present, but ends up
    # being called when the value is _retrieved_, not when it is stored (which
    # never happens now)
    def __getitem__(self, index):
        # simulate the usual dict behavior
        if index not in self.index_set:
            raise KeyError(index)

        # get the value that would have been generated and passed to
        # _setitem_when_not_present for this index
        value = self.rule(self.block, index)

        # the rest is adapted from
        # IndexedComponent._setitem_when_not_present

        # This part is skipped because with delayed constraints, we always
        # have to return a constraint object, unlike _setitem_when_not_present,
        # where obj.set_value(Constraint.Skip) will delete the index entry
        # instead.
        # # If the value is "Skip" do not create an object
        # if value is IndexedComponent.Skip:
        #     return None

        if value is Constraint.Skip:
            # create a constraint object with a valid Pyomo expression
            # instead of skipping it entirely (which we can't do at this stage)
            value = EqualityExpression([0, 0])

        obj = self.component._ComponentDataClass(component=self.component)
        obj._index = index
        obj.set_value(value)

        return obj

    # Override __setitem__ to raise an error (no stored values, have to use the rule)
    # TODO: maybe allow storing a value, then check for that in __getitem__ before
    # calling the rule.
    def __setitem__(self, index, value):
        raise RuntimeError(
            f"{self.component.name} does not support setting items directly."
        )

    def __delitem__(self, index, value):
        raise RuntimeError(
            f"{self.component.name} does not support deleting items directly."
        )

    # Override __iter__ to iterate over the index_set keys
    def __iter__(self):
        return self.keys()

    def __str__(self):
        return f"{self.__class__}({dict(self.items())})"

    def __len__(self):
        return len(self.index_set)  # not quite right, but hopefully good enough

    # Override keys() to return index_set as the keys
    def keys(self, sort=False):
        return (key for key, val in self.items(sort))

    # Override items() to return key, rule(key) pairs
    def items(self, sort=False):
        return ((key, val) for key in self.index_set if (val := self[key]) is not None)

    # Override values() to return rule(key) values for all keys in index_set
    def values(self, sort=False):
        return (val for key, val in self.items(sort))

    # Override __contains__ to check membership in index_set
    def __contains__(self, key):
        return key in self.index_set


# patch Pyomo so Switch will use DelayedExpression instead of Expression
pyomo.core.base.expression.IndexedExpression = DelayedIndexedExpression
pyomo.core.base.constraint.IndexedConstraint = DelayedIndexedConstraint
