"""
Call AMPL NL-style solver with minimal memory overhead.
To use, add this module to the module list or import it, then call any ampl
solver as normal, e.g., --solver coptampl. For solvers with multiple io modes
like gurobi, you may need to also specify --solver-io nl.

To download ampl solvers, you can do this:
```
conda install amplpy
python -m amplpy.modules install <solver name>
```
This will then run a command like
`pip install -i https://pypi.ampl.com ampl_module_base ampl_module_mosek`
Then you need to move the program into your search path, e.g.,
ln -s ~/miniforge3/envs/switch-pg-reeds-local/lib/python3.10/site-packages/ampl_module_mosek/bin/mosek ~/miniforge3/envs/switch-pg-reeds-local/bin/mosek_ampl
Note: the pyomo mosek interface gives up when you specify solver solver-io nl, so you
either need to specify the path to the solver like --solver `which mosek` or else
give the mosek ampl solver a different name like mosek_ampl

See Python instructions at https://portal.ampl.com/user/ampl/amplce/enterprise
for details on available solvers.
For proprietary solvers, you will also need to obtain and install a license.

TODO: handle infeasible or unbounded models

TODO: maybe implement this as a model writer instead of a solver via
@WriterFactory.register('nl', doc). (see notes at end of file)
TODO: handle fixed variables, e.g., switch_model.hawaii.smooth_dispatch (are these
automatically treated as numbers in constraints and omitted from active variable list?)
TODO: this works well for coptampl and probably gurobi but appears to read back
the wrong variable values with cplexamp
TODO: the small license_is_valid() model gives an error reading the .sol file for coptampl,
e.g.,
m = pyo.ConcreteModel()
m.x = pyo.Var(bounds=(1, 2))
m.obj = pyo.Objective(expr=m.x)
SolverFactory('coptampl', solver_io='nl').solve(m)

TODO: change delayed_expressions code to only delay construction of the body of
the constraint, not the bounds? i.e., create the ConstraintData component, but
use the rule to create the expression for the body when needed, instead of
storing it in the body; but keep the bounds? this can make dual lookup possible?
But this will force double-running of the function (e.g., when accessing ub, lb
then body), so maybe better to retrieve a just-in-time-constructed
ConstraintData object once, then access ub, lb and body all together (like we do
now), and maybe also use some kind of hash to lookup the duals? We could instead
define a __hash__ element for the custom ConstraintData component that is either
c.name or (id(c.parent), c.index) or a hash of these.

TODO: Implement parallel generation of constraint data (not easy). See notes at
end of file.
"""

import array
import os
import shutil
import subprocess
import time
import string, uuid, tempfile
from itertools import islice
from typing import List, Tuple, Sequence, Any, Dict

import numpy as np
from scipy import sparse

import pyomo.environ as pyo
from pyomo.repn.standard_repn import generate_standard_repn
from pyomo.opt import SolverFactory
from pyomo.opt.solver import SystemCallSolver
from pyomo.opt.results import SolverResults
from pyomo.opt.results.solver import SolverStatus, TerminationCondition
from pyomo.common.tee import capture_output

inf = float("inf")


# hook in as solver for a Switch model
# def define_components(m):
#     # save external solver name for later
#     m.options.nl_solver = m.options.solver
#     # assign "nl_solver" as the solver for this model
#     m.options.solver = "nl_solver"


def ampl_bound_info(lhs, rhs):
    """
    Return a string indicating the type of bounds for a variable or constraint
    and the numerical value(s) of the finite bound(s).

    These are from Table 17 of https://ampl.github.io/nlwrite.pdf.
    """
    if lhs == -inf:
        if rhs == inf:
            return f"3"  # unbounded
        else:
            return f"1 {rhs}"  # x <= rhs # g
    elif rhs == inf:
        return f"2 {lhs}"  # lhs <= x  # g
    elif rhs == lhs:
        return f"4 {rhs}"  # x = rhs  # g
    else:
        return f"0 {lhs} {rhs}"  # lhs <= x <= rhs  # g


uint_max = [(c, (1 << (array.array(c).itemsize * 8)) - 1) for c in "BHILQ"]


def uint_array_type(num):
    """
    Return the smallest unsigned int array typecode that can contain `num`.
    """
    for c, umax in uint_max:
        if num <= umax:
            return c
    raise ValueError(f"No unsigned int array type can hold {num}")


def unique_name():
    # create a globally unique string of letters and digits
    chars = string.digits + string.ascii_letters
    base = len(chars)
    x = int(uuid.uuid4().hex, 16)  # large, globally unique int
    name = ""
    while x:
        x, digit = divmod(x, base)
        name += chars[digit]
    return name


# Replace standard asl solver with this one; "asl" is automatically used when
# the solver name is not recognized, which covers most cases we want.
@SolverFactory.register("asl", doc="low-memory AMPL NL solver (LP/MIP only)")
# Catch a couple of cases where a known solver will defer to something other
# than simply 'asl' when mode == 'nl'. (This may not be a drop-in replacement,
# but we give it a try.) Note that this is only needed if for example the ampl
# version of gurobi is installed as 'gurobi' somewhere on the search path. More
# often the user may need to specify the path to find it or the executable will
# be called gurobiasl or gurobi_ampl. In any of these cases, the asl solver will
# be used automatically.
@SolverFactory.register("_gurobi_nl", doc="low-memory AMPL NL solver (LP/MIP only)")
@SolverFactory.register("_cbc_shell", doc="low-memory AMPL NL solver (LP/MIP only)")
class NLSolver(SystemCallSolver):
    """
    Minimal Pyomo solver plugin that calls an AMPL solver to solve continuous
    linear programs (LPs).

    The key difference from Pyomo's built-in ASL solver is that this does not
    retain references to elements of Constraint or Expression components, so it
    can work with the delayed_expressions to sharply reduce memory requirements
    (though it may run a little slower due to regenerating expressions each time
    they are needed).

    Limitations:
        * All decision variables must be continuous (no integer/binary).
        * The objective and all constraints must be linear.
        * Quadratic or general nonlinear expressions are rejected.
        * Has not been robustly tested with multiple solvers (works OK with COPT).
    """

    def __init__(self, **kwds):
        # based on pyomo.solvers.plugins.solvers.ASL.ASL
        if not "type" in kwds:
            kwds["type"] = "asl"
        SystemCallSolver.__init__(self, **kwds)

    # def __init__(self, **kwds):
    #     self._options = dict()

    # # required by GenericSolverInterface pattern
    # @property
    # def options(self):
    #     return self._options

    # @options.setter
    # def options(self, val):
    #     self._options = dict(val)

    # def available(self, exception_flag=True):
    #     return True

    # def license_is_valid(self):
    #     return self.available(exception_flag=False)

    def version(self):
        return (0, 0, 0, 0)

    # generic model test from pyomo.solvers.plugins.solvers.GUROBI.GUROBINL
    def license_is_valid(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(1, 2))
        m.obj = pyo.Objective(expr=m.x)
        try:
            with capture_output():
                self.solve(m)
                return abs(m.x.value - 1) <= 1e-4
        except:
            return False

    # ---- main entry point ----
    def solve(
        self,
        model,
        tee=False,
        load_solutions=True,
        logfile=None,
        solnfile=None,
        timelimit=None,
        report_timing=False,
        solver_io=None,
        suffixes=None,
        options=None,
        keepfiles=False,
        symbolic_solver_labels=False,
    ):
        """
        Solve a Pyomo model using an NL-based solver.

        This routine writes primal variable values directly back to `model`.
        It returns a minimal SolverResults object with solver status only.
        """

        # todo: use Pyomo methods to create/remove temp dir and name .nl file,
        # as discussed under "model writer" notes below

        tempdir = model.options.tempdir
        if tempdir:
            keepfiles = True
        else:
            tempdir = tempfile.mkdtemp()
            keepfiles = bool(model.options.keepfiles)

        file_stub = os.path.join(tempdir, unique_name())
        nl_file = file_stub + ".nl"
        sol_file = file_stub + ".sol"

        t_start = time.time()

        print(f"nl_solver: will create nl_file {nl_file}", flush=True)
        model_info = write_nl_file(model, nl_file)

        # Gather options together
        # opt = dict(self._options)
        # if options is not None:
        #     opt.update(options)
        opt = options or {}

        out_handle = None if tee else subprocess.DEVNULL

        # convert options to suitable form for AMPL solver
        solver_args = [
            str(k) + ("" if v is True or v == "" else f"={v}") for k, v in opt.items()
        ]

        print("nl_solver: calling solver", flush=True)
        if os.path.exists(sol_file):
            os.remove(sol_file)

        args = [self.options.solver, file_stub, "-AMPL"] + solver_args
        subprocess.run(
            args,
            check=True,
            stdout=out_handle,
            stderr=out_handle,
        )

        if not os.path.exists(sol_file):
            raise RuntimeError(f"Solver did not produce solution file.")

        # write solution to Pyomo vars in same order as sent to solver
        if load_solutions:
            print(f"nl_solver: loading solution into model", flush=True)
            solver_msg = process_sol_file(model, sol_file, model_info)
            # take a guess if we got this far
        else:
            solver_msg = ""
            term_cond = TerminationCondition.error
            solver_status = SolverStatus.error

        # assume it worked out if we got this far (the .sol file doesn't seem
        # to contain info on errors, so it will probably just be unreadable
        # for unsolved models)
        term_cond = TerminationCondition.optimal
        solver_status = SolverStatus.ok

        # ---- build a minimal SolverResults object -----------------------------
        print(f"nl_solver: preparing results object", flush=True)
        results = SolverResults()
        results.solver.name = "nl_solver"
        results.solver.status = solver_status
        results.solver.termination_condition = term_cond
        results.solver.message = solver_msg
        results.problem.name = model.name
        results.problem.number_of_variables = model_info["variable_count"]
        results.problem.number_of_constraints = model_info["constraint_count"]
        results.problem.sense = model_info["objective_sense"]
        results.solver.wallclock_time = time.time() - t_start

        print(f"nl_solver: returning results: {results}", flush=True)
        if not keepfiles:
            shutil.rmtree(tempdir)
        return results


def header_str(model_info, default=None):
    def val(key):
        v = model_info.get(key, default)
        if v is None:
            raise ValueError(f"No value provided for {key}")
        else:
            return v

    header_data = [
        # format, ampl internal, ampl internal, ampl internal
        ["g3", 1, 1, 0],
        # *vars, *constraints, *objectives, *ranges, *eqns, lcons
        [
            val("variable_count"),
            val("constraint_count"),
            val("objective_count"),
            val("constraint_range_count"),
            val("constraint_equation_count"),
            # 0,
        ],
        # nonlinear constraints, objectives
        [0, 0] + [0, 0, 0, 0],
        # network constraints: nonlinear, linear
        [0, 0],
        # nonlinear vars in constraints, objectives, both
        [0, 0, 0],
        # linear network variables; functions; arith, flags
        [0, 0, 0, 1],
        # discrete variables: *binary, *integer, nonlinear (b,c,o)
        [val("binary_variable_count"), val("integer_variable_count"), 0, 0, 0],
        # nonzeros in Jacobian, gradients
        [val("constraint_coefficient_count"), val("objective_coefficient_count")],
        # max name lengths in .row and .col files: constraints, variables
        [0, 0],
        # common exprs: b, c, o, c1, o1
        [0, 0, 0, 0, 0],
    ]

    lines = [" ".join(str(v) for v in row) for row in header_data]
    for i in range(1, len(lines)):
        # add leading space, since that seems to be standard
        lines[i] = " " + lines[i]
    lines.append("")  # force creation of final "\n"

    return "\n".join(lines)


def get_int(f):
    return int(f.readline().strip())


def get_float(f):
    return float(f.readline().strip())


def write_nl_file(model, file):
    # This version is optimized to minimize in-memory data structures, but can access
    # constraints and objectives repeatedly.

    # create empty model_info dict, which will be used to create a dummy
    # header, then filled in as the file is written, then used to create
    # the real header
    model_info = {}

    print("nl_solver: building variable list", flush=True)
    model_info["binary_variable_count"], model_info["integer_variable_count"], vars = (
        vars_for_solver(model)
    )

    var_num = {id(v): i for i, v in enumerate(vars)}
    var_count = len(var_num)
    if var_count == 0:
        raise ValueError("nl_solver: model has no active variables.")
    model_info["variable_count"] = var_count

    # Gather constraint data for use for the constraint counts, r segments
    # and c segments. It would be nice to avoid holding so much info in RAM
    # (and it's not clear how fast Python will give it back), but this
    # should be a lot faster than constructing the constraint expressions
    # twice, once for the r segments and once for the c segments. We use
    # arrays to keep data requirements as tight as possible. If this is still
    # too much (and the memory isn't released when these are deleted), we could
    # delegate all of this to a secondary process that releases its memory
    # when finished.
    # For a model with 675k rows and 4.9M nonzeroes (4.1 GB in COPT),
    # this will take approx. 13.5 MB total for the lhs, rhs and var_cnt
    # arrays and 59 MB for the var_num and var_coeff arrays.
    # lower bound for constraint n
    c_lhs = array.array("d")
    # upper bound for constraint n
    c_rhs = array.array("d")
    # number of vars referenced in constraint n (for the array data type, we
    # assume in worst case all variables in the problem could appear in one
    # constraint)
    c_term_count = array.array(uint_array_type(var_count - 1))
    # solver id numbers of vars in each constraint (in sequence)
    c_var_num = array.array(uint_array_type(var_count - 1))
    # coefficients of vars in each constraint (in sequence)
    c_var_coeff = array.array("d")

    # Creating the repn is expensive (approx. as long as generating
    # the expressions), and we need it for both the
    # constraint bounds and the constraint matrix, so we create
    # it once and store for use for both, even though that takes
    # some memory
    # TODO: test increase in time if we generated the repn twice
    # TODO: figure out how much memory we are willing to trade for
    # how much time.
    # TODO: maybe it's OK to hold temporary largish structures while writing the
    # .sol file because we'll give back the memory before calling the solver, so
    # they won't affect our ability to solve big models.
    # TODO: therefore, access each component only once, including the objective,
    # which gives us the flexibility to use delayed generation if we want to
    # further minimize memory, or use full contruction so we can get duals, etc.
    print("nl_solver: building constraint list", flush=True)
    for i, c in enumerate(
        model.component_data_objects(pyo.Constraint, active=True, descend_into=True)
    ):
        repn = generate_standard_repn(c.body, quadratic=False)
        if not repn.is_linear():
            raise TypeError(
                "nl_solver only supports linear constraints. "
                f"Constraint {c.name} is not linear."
            )
        const = repn.constant or 0.0
        c_lhs.append((float(pyo.value(c.lower)) if c.has_lb() else -inf) - const)
        c_rhs.append((float(pyo.value(c.upper)) if c.has_ub() else inf) - const)

        c_term_count.append(len(repn.linear_coefs))
        c_var_num.extend(var_num[id(x)] for x in repn.linear_vars)
        c_var_coeff.extend(float(x) for x in repn.linear_coefs)

    model_info["constraint_count"] = len(c_term_count)
    model_info["constraint_range_count"] = sum(
        1 for lhs, rhs in zip(c_lhs, c_rhs) if lhs != rhs and lhs > -inf and rhs < inf
    )
    model_info["constraint_equation_count"] = sum(
        1 for lhs, rhs in zip(c_lhs, c_rhs) if lhs == rhs
    )
    model_info["constraint_coefficient_count"] = len(c_var_coeff)

    # gather and count objective terms
    print("nl_solver: gathering objective terms", flush=True)
    # number of vars referenced in objective n (for the array data type, we
    # assume in worst case all variables in the problem could appear in one
    # objective)
    o_term_count = array.array(uint_array_type(var_count - 1))
    # optimization sense (pyo.minimize or pyo.maximize) for each objective
    o_sense = []
    # constant term in objective
    o_constant = []
    # solver id numbers of vars in each objective (in sequence)
    o_var_num = array.array(uint_array_type(var_count - 1))
    # coefficients of vars in each objective (in sequence)
    o_var_coeff = array.array("d")
    for i, o in enumerate(
        model.component_data_objects(pyo.Objective, active=True, descend_into=True)
    ):
        repn = generate_standard_repn(o.expr, quadratic=False)
        if not repn.is_linear():
            raise TypeError(
                "nl_solver only supports linear objectives. "
                "Quadratic / nonlinear terms are not allowed."
            )
        o_constant.append(repn.constant)
        o_term_count.append(len(repn.linear_coefs))
        o_var_num.extend(var_num[id(x)] for x in repn.linear_vars)
        o_var_coeff.extend(float(x) for x in repn.linear_coefs)
        o_sense.append(o.sense)

    del o, repn
    model_info["objective_count"] = len(o_term_count)
    model_info["objective_coefficient_count"] = len(o_var_coeff)
    # use the optimization sense from the last active objective for reporting
    model_info["objective_sense"] = o_sense[-1]

    print(f"nl_solver: writing NL file {file}", flush=True)
    with open(file, "w") as nl:
        nl.write(header_str(model_info))

        # stream model components into the .nl file in the order they occur;
        # we will read results back in this same order, so we don't need to create a
        # mapping

        # C segments: algebraic constraint bodies’ nonlinear parts; for now we
        # set all these to zero (only allow linear constraints)
        for i in range(len(c_term_count)):
            nl.write(f"C{i}\n")
            nl.write("n0\n")

        # O segments: objective nonlinear part (0) plus sense flag sigma (0=min, 1=max)
        for i, (sense, constant) in enumerate(zip(o_sense, o_constant)):
            nl.write(f"O{i} {0 if sense == pyo.minimize else 1}\n")
            nl.write(f"n{constant}\n")

        # x segment: existing primal values for variables (TODO)
        nl.write("x0\n")

        # r segment: bounds on algebraic constraint bodies (“ranges” / rhs)
        nl.write("r\n")
        for lhs, rhs in zip(c_lhs, c_rhs):
            nl.write(ampl_bound_info(lhs, rhs) + "\n")

        # b segment: variable bounds
        nl.write("b\n")
        for v in vars:
            lb = -inf if v.lb is None else v.lb
            ub = inf if v.ub is None else v.ub
            nl.write(ampl_bound_info(lb, ub) + "\n")

        # k segment: Jacobian cumulative column counts, must precede all J segments
        if var_count > 1:
            col_counts = array.array(uint_array_type(var_count - 1), [0]) * var_count
            for n in c_var_num:
                col_counts[n] += 1
            nl.write(f"k{var_count - 1}\n")
            running_total = 0
            for n in range(var_count - 1):
                running_total += col_counts[n]
                nl.write(f"{running_total}\n")
            del col_counts
        else:
            nl.write("k0\n")

        # J segments: linear terms in each constraint body
        # note: we take variable numbers and coefficients from their respective
        # arrays in var_cnt-size chunks, one chunk for each constraint
        c_var_data = iter(zip(c_var_num, c_var_coeff))
        for i, term_count in enumerate(c_term_count):
            if term_count > 0:
                # we skip any with no terms because solvers don't like J0 lines but don't
                # mind missing J segments
                nl.write(f"J{i} {term_count}\n")
                # sort terms by var_num for more consistency with Pyomo NL writer
                for var_num, coeff in sorted(islice(c_var_data, term_count)):
                    nl.write(f"{var_num} {coeff}\n")  # g

        # G segment: objective linear terms
        o_var_data = iter(zip(o_var_num, o_var_coeff))
        for i, term_count in enumerate(o_term_count):
            nl.write(f"G{i} {term_count}\n")
            # sort terms by var_num for more consistency with Pyomo NL writer
            # and because typical NL solvers crash if these aren't sorted.
            for var_num, coeff in sorted(islice(o_var_data, term_count)):
                nl.write(f"{var_num} {coeff}\n")  # g

    # return model stats for use by process_sol_file
    return model_info


def vars_for_solver(model):
    """
    Create a list of vars in correct order for an ampl solver; return number of
    binary vars, number of integer vars, list of all vars (as Pyomo components)

    Solver order is continuous, binary, integer, based on Table 3 of
    https://ampl.com/wp-content/uploads/Hooking-Your-Solver-to-AMPL-by-David-M.-Gay.pdf
    or https://www.ampl.com/_archive/first-website/REFS/hooking2.pdf
    """
    bin_vars = []
    int_vars = []
    cont_vars = []
    for v in model.component_data_objects(pyo.Var, active=True, descend_into=True):
        if v.is_binary():
            bin_vars.append(v)
        elif v.is_integer():
            int_vars.append(v)
        else:
            cont_vars.append(v)

    bin_count = len(bin_vars)
    int_count = len(int_vars)
    vars = []
    vars.extend(cont_vars)
    vars.extend(bin_vars)
    vars.extend(int_vars)
    return (bin_count, int_count, vars)


def process_sol_file(model: pyo.ConcreteModel, sol_file: str, model_info: dict) -> str:
    # based on https://github.com/ampl/mp/blob/a9264b84a5fb0009bd098164bbf1e0aecf7e0592/nl-writer2/include/mp/sol-reader2.hpp#L108
    with open(sol_file, "r") as f:
        # get solver message:
        msg = ""
        for row in f:
            if row.strip():
                msg += row
            else:
                # blank line = end of message
                break

        # skip over any additional blank lines
        while True:
            next_row = f.readline().strip()
            if next_row:
                break

        # optionally followed by an "Options" line plus some option values
        if next_row == "Options":
            # structure is as follows:
            # Options
            # 0: (vbtol present) ? n_opt_lines + 2 : n_opt_lines, int (3-9)
            # 1: ?, int
            # 2: (vbtol present) ? 3 : ?, int
            # ... additional ints if n_opt_lines > 3
            # n_opt_lines: ?, int
            # z = n_opt_lines + 1: ?, int
            # z+1: num_cons, int
            # z+2: ?, int
            # z+3: num_vars, int
            # z+4: ?, int
            # vbtol, float (if present)

            options = dict()
            # get first 4 option values, always present
            for i in range(4):
                options[i] = get_int(f)
            assert options[0] >= 3, ".sol file has fewer option values than expected"
            if options[2] == 3:
                # vbtol flag: if set to 3, there will be 2 fewer ints but then
                # there will be a float vbtol value at the end
                n_opt_lines = options[0] - 2
            else:
                n_opt_lines = options[0]

            # read remaining Options data
            for i in range(4, n_opt_lines + 5):
                options[i] = get_int(f)
            # read vbtol if present
            if options[2] == 3:
                vbtol = get_float(f)

            # check that the supplied number of vars and constraints matches
            # what is declared in the file
            z = n_opt_lines  # + 1 # orig code includes + 1 but that doesn't seem to work
            assert (
                options[z + 1] == model_info["constraint_count"]
            ), "Unexpected number of constraints found in .sol file"
            assert (
                options[z + 3] == model_info["variable_count"]
            ), "Unexpected number of variables found in .sol file"

        else:
            raise NotImplementedError("Missing code for .sol file with no Options line")

        # read/discard constraint duals (only present if you ask for duals?)
        # for i in range(model_info["constraint_count"]):
        #     get_float(f)

        # read variable values
        bin_count, int_count, vars = vars_for_solver(model)
        for v in vars:
            val = get_float(f)
            lb = v.lb
            ub = v.ub
            # fix small out-of-bounds errors
            if lb is not None and lb - 1e-9 <= val < lb:
                val = lb
            elif ub is not None and ub < val <= ub + 1e-9:
                val = ub
            v.set_value(val)

        # ignore the rest of the file, if any

    return msg


"""
NOTES ON CONVERTING TO MODEL WRITER
To run as a model writer registered via @WriterFactory.register('nl', doc), this
just needs to implement a __call__ method with signature
(filename, smap) = problem_writer(self, filename, solver_capability, io_options)
(see pyomo.core.base.block.BlockData.write() for call.)
smap can probably be None (worth trying anyway), and will be passed to the .sol
reader later.

Will that enable all the standard filename stuff
without doing any standard symbolmap stuff?

pyomo.opt.base.solvers.OptSolver.solve orchestrates most of the interesting stuff in a
solve, e.g.,
# create .nl file and self._smap_id (symbol_map)
self._presolve(*args, **kwds)   # SystemCallSolver(OptSolver)
# call solver and get result code (_status.rc)
_status = self._apply_solver()   # SystemCallSolver(OptSolver)
# read the .sol file into result object
result = self._postsolve()   # SystemCallSolver(OptSolver)
# load the result into the model's solution list and activate it (i.e., assign
# values to the model). Note that result._smap is None at this point, so it
_model.solutions.load_from(result, ...)

### produce .nl file, attach symbol map to model.solutions
# Pyomo.opt.base.solvers.OptSolver._presolve (called by SystemCallSolver._presolve)
(self._problem_files, self._problem_format, self._smap_id) = (
    self._convert_problem(
        args, self._problem_format, self._valid_problem_formats, **kwds
    )
)

# Pyomo.opt.base.solvers.OptSolver._convert_problem finds the right converter
# by checking which ones can convert from the allowed types to the required type,
# probably ProblemConverterFactory('pyomo'), i.e.
# pyomo.solvers.plugins.converter.model.PyomoMIPConverter(). Then it calls
problem_files, symbol_map = converter.apply(*tmp, **tmpkw)
return problem_files, ptype, symbol_map
# note: symbol_map should be symbol_map_id here for consistency with code above and below

# In PyomoMIPConverter.apply():
(problem_filename, symbol_map_id) = instance.write(
    filename=problem_filename,  # tempfile ending in .pyomo.nl
    format=args[1],             # ProblemFormat.nl
    solver_capability=capabilities, # kwarg
    io_options=io_options,          # unused kwargs
)
return (problem_filename,), symbol_map_id

# In pyomo.core.base.block.BlockData.write():
# ***** replace problem_writer with our writer
problem_writer = WriterFactory(format)
(filename, smap) = problem_writer(self, filename, solver_capability, io_options)
smap_id = id(smap)
self.solutions = ModelSolutions(self)
self.solutions.add_symbol_map(smap)
return filename, smap_id

# Entirety of pyomo.core.base.PyomoModel.ModelSolutions.add_symbol_map(self, symbol_map)):
self.symbol_map[id(symbol_map)] = symbol_map

#### run solver
# pyomo.opt.solver.shellcmd.SystemCallSolver._apply_solver:
# self._command is previously set by SystemCallSolver._presolve
self._rc, self._log = self._execute_command(self._command)


#### read .sol file, use model.solutions.symbol_map
# pyomo.opt.solver.shellcmd.SystemCallSolver._postsolve(self):
results = self.process_output(self._rc)   # self._rc was stored by self._apply_solver()

# pyomo.opt.solver.shellcmd.SystemCallSolver.process_output(self, rc):
# finds the right reader for this type of file
"""

"""
NOTES ON PARALLEL GENERATION OF CONSTRAINT DATA
This will not be easy because process forking is not available on Windows and is
considered unsafe on macOS. So spawning may be needed instead. But spawning
requires pickling any context data to pass it to the process, and Pyomo models
(specifically the lambdas we use for rules, but maybe other elements) can't be
pickled to do this. So we'd have to build the model and variable list separately
in each worker as background data, then call the workers with constraint names
and indices to get chunks of data to add to the constraint arrays. That should
work, especially since model construction is quite fast with the delayed
expressions and constraints. But it will be tough to setup and will depend on
sharp determinism between the different models (i.e., constraint and var
elements created in exactly the same order in each one). It will also require
extra memory for each process, so we'll probably want to squeeze the model
memory down further, e.g., with delayed generation of the objective expression.

As a starting point for parallel constraint processing, you could use `for comp
in model.component_objects(pyo.Constraint, active=True, descend_into=True):`
instead of iterating over `model.component_data_objects(pyo.Constraint,
active=True, descend_into=True)`, then generate constraint names and indices
using code similar to the body of
pyomo.core.base.block.BlockData._component_data_itervalues(), which is called by
BlockData.component_data_objects. The PseudoMap() iterator in that function is
pretty equivalent to `model.component_objects()`. (Workers will need to accept
the constraint name and indices and return corresponding lhs, rhs, term_count
values and var_num, var_coeff lists.)
"""
