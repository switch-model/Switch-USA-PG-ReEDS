import os
from pyomo.environ import *
from switch_model.utilities import unique_list


def define_arguments(argparser):
    argparser.add_argument(
        "--find-current-prm",
        action="store_true",
        default=False,
        help="Instead of minimizing costs, find the highest planning reserve margin that can be applied across all load zones in all timepoints.",
    )
    argparser.add_argument(
        "--maximize-prm-sum",
        action="store_true",
        default=False,
        help="Instead of minimizing costs, find the set of planning reserve margins for each load zone and timeseries whose sum is as high as possible. This will be done in addition to any pre-specified planning reserve margin.",
    )


def define_components(m):
    m.PR_ZONE_TS = Set(dimen=2, within=m.LOAD_ZONES * m.TIMESERIES)
    m.PR_ZONE_TPS = Set(
        dimen=2,
        initialize=lambda m: [
            (z, tp) for (z, ts) in m.PR_ZONE_TS for tp in m.TPS_IN_TS[ts]
        ],
    )
    m.planning_reserve_margin = Param(m.PR_ZONE_TS, within=Reals)

    if m.options.find_current_prm:
        # maximize a single PRM across all zones (to find the maximum feasible
        # PRM under current conditions)
        m.MaxPRMargin = Var(within=Reals)  # will become model objective in pre_solve()
        m.CurrentPRMargin = Expression(
            m.PR_ZONE_TS, rule=lambda m, z, ts: m.MaxPRMargin
        )
        prm_var = m.CurrentPRMargin

    elif m.options.maximize_prm_sum:
        m.PR_TS = Set(initialize=lambda m: unique_list(ts for z, ts in m.PR_ZONE_TS))
        m.MaxPRMarginTS = Var(m.PR_TS, within=Reals)
        m.MaxPRMargin = Expression(
            m.PR_ZONE_TS, rule=lambda m, z, ts: m.MaxPRMarginTS[ts]
        )

        m.Respect_Standard_PRM = Constraint(
            m.PR_ZONE_TS,
            rule=lambda m, z, ts: m.MaxPRMargin[z, ts]
            >= m.planning_reserve_margin[z, ts],
        )
        # find largest possible total reserve margin across zones
        # on a MW basis (useful for seeing where/when the reserve margin
        # is binding and where it isn't)
        # m.PRMSum = Expression(  # will become model objective in pre_solve()
        #     rule=lambda m: sum(
        #         m.MaxPRMargin[z, ts] * m.zone_demand_mw[z, tp]
        #         for (z, ts) in m.PR_ZONE_TS
        #         for tp in m.TPS_IN_TS[ts]
        #     )
        # )
        # will become model objective in pre_solve()
        m.PRMSum = Expression(rule=lambda m: sum(m.MaxPRMarginTS[ts] for ts in m.PR_TS))
        prm_var = m.MaxPRMargin
    else:
        prm_var = m.planning_reserve_margin

    if m.options.find_current_prm or m.options.maximize_prm_sum:

        def rule(m):
            if len(m.PR_ZONE_TS) == 0:
                raise ValueError(
                    "This module reauires reserve margin zones and timeseries "
                    "to be defined in planning_reserve_margin.csv."
                )

        m.Check_for_PR_ZONE_TS = BuildAction(rule=rule)

    # have to use name to survive transfer to model instance
    prm_var_name = prm_var.name
    m.planning_reserves = Expression(
        m.LOAD_ZONES,
        m.TIMEPOINTS,
        rule=lambda m, z, tp: (
            # apply the planning reserve margin for the timeseries that holds this timepoint
            getattr(m, prm_var_name)[z, m.tp_ts[tp]] * m.zone_demand_mw[z, tp]
            if (z, tp) in m.PR_ZONE_TPS
            else 0
        ),
    )
    # add planning reserves in the same location as loads
    try:
        m.Distributed_Power_Withdrawals.append("planning_reserves")
    except AttributeError:
        m.Zone_Power_Withdrawals.append("planning_reserves")


def pre_solve(m):
    if m.options.find_current_prm:
        # deactivate current objective and maximize MaxPRMargin instead
        for o in m.component_objects(Objective, active=True):
            o.deactivate()
        m.Maximize_PRMargin = Objective(rule=lambda m: m.MaxPRMargin, sense=maximize)
    elif m.options.maximize_prm_sum:
        # deactivate current objective and maximize PRMSum instead
        for o in m.component_objects(Objective, active=True):
            o.deactivate()
        m.Maximize_PRMargin = Objective(rule=lambda m: m.PRMSum, sense=maximize)


def post_solve(m, outputs_dir):
    if m.options.find_current_prm:
        m.logger.info(
            "\nSolved for highest attainable planning reserve margin shared across all regions:"
            f"\n{value(m.MaxPRMargin)}"
            "\nNOTE: the `--find-current-prm` flag was used for this run. This maximizes the "
            "\nplanning reserve margin, ignoring financial costs. Results from this run other "
            "\nthan the reserve margin calculation should be ignored.\n"
        )
    elif m.options.maximize_prm_sum:
        m.logger.info(
            "\nSolved for highest attainable sum of planning reserve margins across all timeseries."
            f"\nSee {os.path.join(outputs_dir, 'MaxPRMarginTS.csv')} for values for timeseries."
            "\nNOTE: the `--maximize-prm-sum` flag was used for this run. This maximizes planning"
            "\nreserve margins, ignoring financial costs. Results from this run other than the "
            "\nreserve margin values should be ignored.\n"
        )


def load_inputs(model, switch_data, inputs_dir):
    """
    Files or columns marked with * are optional. See notes above on default
    values.

    planning_reserve_margin.csv*
        LOAD_ZONE, TIMESERIES, planning_reserve_margin*
    """
    switch_data.load_aug(
        filename=os.path.join(inputs_dir, "planning_reserve_margin.csv"),
        optional=True,
        index=model.PR_ZONE_TS,
        param=(model.planning_reserve_margin,),
    )
