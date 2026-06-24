"""
Applies investment tax credit (ITC) to up to the specified number of MW of
generators that use the specified fuel source and are built in the specified
date range. This is useful for modeling a general ITC or allocation of
safe-harbor ITC to a certain amount of capacity built in the specified date
range.

ITC is applied first to predetermined/planned generators, prorated equally if
planned capacity exceeds the allowed amount of ITC. Then any remaining ITC MW
are applied to model-selected, "free" generators. Distributed generators are
excluded from ITC.

The assignment to model-selected generators uses a two-stage approach. During
the main solve, Switch is free to choose which generators get the ITC,
subject to the limit on total number of MW receiving ITC. Then in the post-solve
stage, we reassign the ITC equally to all eligible generators that were built.
This reduces the tendency to assign ITC only to the most expensive of the
generators that are built.

We can't apply the exact ITC endogenously in the model because there are only
two ways to do this, both bad: (1) let the model decide which gens will get the
ITC, in which case it will inaccurately apply it to only the most expensive
eligible gens that get built, or (2) apply ITC endogenously based on average
capital recovery of gens that get built; this requires a nonlinear expression
that can't be included in a linear program.
"""

import os, bisect
from pyomo.environ import *
from switch_model.financials import capital_recovery_factor as crf

inf = float("inf")


def define_components(m):
    """
    TODO
    """

    # List of ITC definitions, specified as tuples of energy source, start year,
    # end year (es, s, e), indicating which source is eligible and what years
    # gens must be built to qualify
    m.ITC_GROUPS = Set(dimen=3)

    # investment tax credit rate for each ITC group (fraction of capital cost)
    m.itc_rate = Param(m.ITC_GROUPS, within=PercentFraction)
    # maximum MW that can receive ITC in each group (omit if unlimited)
    m.itc_eligible_mw = Param(m.ITC_GROUPS, within=NonNegativeReals, default=inf)

    # generator/build-year combinations that are eligible for each ITC group
    m.GEN_BLD_YRS_FOR_ITC_GROUP = Set(m.ITC_GROUPS, within=m.GEN_BLD_YRS)

    def rule(m):
        # s = first year of eligibility
        # e = last year of eligibility (e.g., plants built in year e will qualify)
        for es, s, e in m.ITC_GROUPS:
            for g in m.GENS_BY_ENERGY_SOURCE[es]:
                for by in m.BLD_YRS_FOR_GEN[g]:
                    if s <= by <= e and not m.gen_is_distributed[g]:
                        m.GEN_BLD_YRS_FOR_ITC_GROUP[es, s, e].add((g, by))

    m.Build_GEN_BLD_YRS_FOR_ITC_GROUP = BuildAction(rule=rule)

    # Split eligible gens into predetermined and free groups
    m.PREDET_GEN_BLD_YRS_FOR_ITC_GROUP = Set(m.ITC_GROUPS)
    m.FREE_GEN_BLD_YRS_FOR_ITC_GROUP = Set(m.ITC_GROUPS)

    def rule(m):
        for es, s, e in m.ITC_GROUPS:
            for g, by in m.GEN_BLD_YRS_FOR_ITC_GROUP[es, s, e]:
                if (g, by) in m.PREDETERMINED_GEN_BLD_YRS:
                    m.PREDET_GEN_BLD_YRS_FOR_ITC_GROUP[es, s, e].add((g, by))
                else:
                    m.FREE_GEN_BLD_YRS_FOR_ITC_GROUP[es, s, e].add((g, by))

    m.Build_PREDET_FREE_GEN_BLD_YRS_FOR_ITC_GROUP = BuildAction(rule=rule)

    # all valid combinations of ITC group and gen build year
    m.ITC_GROUP_GEN_BLD_YRS = Set(
        dimen=5,
        within=m.ITC_GROUPS * m.GEN_BLD_YRS,
        initialize=lambda m: [
            (es, s, e, g, by)
            for (es, s, e) in m.ITC_GROUPS
            for g, by in m.GEN_BLD_YRS_FOR_ITC_GROUP[es, s, e]
        ],
    )

    # amount of capacity of each gen addition that will receive ITC (MW)
    m.ApplyITCtoGenBuild = Var(m.ITC_GROUP_GEN_BLD_YRS, within=NonNegativeReals)

    # credited capacity can't exceed amount constructed
    m.Max_ApplyITCtoGenBuild = Constraint(
        m.ITC_GROUP_GEN_BLD_YRS,
        rule=lambda m, es, s, e, g, by: m.ApplyITCtoGenBuild[es, s, e, g, by]
        <= m.BuildGen[g, by],
    )

    # Amount of predetermined gen capacity eligible to receive ITC
    m.itc_group_predet_eligible_mw = Param(
        m.ITC_GROUPS,
        rule=lambda m, es, s, e: sum(
            m.build_gen_predetermined[g, by]
            for (g, by) in m.PREDET_GEN_BLD_YRS_FOR_ITC_GROUP[es, s, e]
        ),
    )
    # Apply predet ITC either to all of the eligible predetermined gens or the
    # max ITC capacity, whichever is smaller
    m.itc_group_predet_mw = Param(
        m.ITC_GROUPS,
        rule=lambda m, es, s, e: min(
            m.itc_group_predet_eligible_mw[es, s, e],
            m.itc_eligible_mw[es, s, e],
        ),
    )
    # Allocate ITC in equal shares to all predetermined generators
    m.Prorate_ITC_Predet = Constraint(
        m.ITC_GROUP_GEN_BLD_YRS,
        rule=lambda m, es, s, e, g, by: (
            (
                m.ApplyITCtoGenBuild[es, s, e, g, by]
                == m.build_gen_predetermined[g, by]
                * m.itc_group_predet_mw[es, s, e]
                / m.itc_group_predet_eligible_mw[es, s, e]
            )
            if (g, by) in m.PREDETERMINED_GEN_BLD_YRS
            else Constraint.Skip
        ),
    )

    # Allocation to predetermined and free generators can't exceed total
    # capacity allowed for ITC
    m.Max_Free_ApplyITCtoGenBuild = Constraint(
        m.ITC_GROUPS,
        rule=lambda m, es, s, e: (
            (
                sum(
                    m.ApplyITCtoGenBuild[es, s, e, g, by]
                    for g, by in m.FREE_GEN_BLD_YRS_FOR_ITC_GROUP[es, s, e]
                )
                <= m.itc_eligible_mw[es, s, e] - m.itc_group_predet_mw[es, s, e]
            )
            if m.FREE_GEN_BLD_YRS_FOR_ITC_GROUP[es, s, e]
            else Constraint.Skip
        ),
    )

    # Identify how much ITC-eligible capacity is active in each period,
    # so we can apply the ITC on a per-period basis
    # p: [(g, by, es, s, e)]
    m.GEN_BLD_YR_ITC_GROUPS_ACTIVE_IN_PERIOD = Set(m.PERIODS)

    def rule(m):
        for es, s, e in m.ITC_GROUPS:
            for g, by in m.GEN_BLD_YRS_FOR_ITC_GROUP[es, s, e]:
                for p in m.PERIODS_FOR_GEN_BLD_YR[g, by]:
                    m.GEN_BLD_YR_ITC_GROUPS_ACTIVE_IN_PERIOD[p].add((g, by, es, s, e))

    m.build_GEN_BLD_YR_ITC_GROUPS_ACTIVE_IN_PERIOD = BuildAction(rule=rule)

    # apply the appropriate rate to all capacity selected for ITC
    def rule(m, p):
        return (-1) * sum(
            m.ApplyITCtoGenBuild[es, s, e, g, by]
            * m.itc_rate[es, s, e]
            * m.gen_capital_cost_annual[g, by]
            for (g, by, es, s, e) in m.GEN_BLD_YR_ITC_GROUPS_ACTIVE_IN_PERIOD[p]
        )

    m.GenITC = Expression(m.PERIODS, rule=rule)

    # add the credit to the model objective function
    m.Cost_Components_Per_Period.append("GenITC")


def post_solve(m, outdir):
    # update the ITC allocation for free gens to apply an equal share of ITC to
    # all eligible free gens that were built, instead of letting the model
    # preferentially apply it to the most expensive ones
    old_system_cost = value(m.SystemCost)
    for es, s, e in m.ITC_GROUPS:
        total_itc_used = sum(
            value(m.ApplyITCtoGenBuild[es, s, e, g, by])
            for (g, by) in m.FREE_GEN_BLD_YRS_FOR_ITC_GROUP[es, s, e]
        )
        total_built = sum(
            value(m.BuildGen[g, by])
            for (g, by) in m.FREE_GEN_BLD_YRS_FOR_ITC_GROUP[es, s, e]
        )
        itc_share = 0 if total_built == 0 else (total_itc_used / total_built)
        for g, by in m.FREE_GEN_BLD_YRS_FOR_ITC_GROUP[es, s, e]:
            m.ApplyITCtoGenBuild[es, s, e, g, by] = itc_share * m.BuildGen[g, by]
    new_system_cost = value(m.SystemCost)
    m.logger.info(
        f"System cost changed by equal allocation of ITC to new-build gens: "
        f"{old_system_cost:,.0f} -> {new_system_cost:,.0f}"
    )


def load_inputs(m, switch_data, inputs_dir):
    """
    Import ITC eligibility and rate data.

    itc.csv
        ENERGY_SOURCE, START_YEAR, END_YEAR, itc_rate, [itc_eligible_mw]

    """

    switch_data.load_aug(
        optional=True,
        filename=os.path.join(inputs_dir, "itc.csv"),
        index=m.ITC_GROUPS,
        param=(m.itc_rate, m.itc_eligible_mw),
    )
