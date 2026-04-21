"""
Add an UnservedLoad component, which ensures the model is always feasible.
This is often useful when the model is constrained to the edge of infeasibility,
(e.g., when evaluating a pre-defined, just-feasible construction plan) to avoid
spurious reports of infeasibility.

Based on switch_model.hawaii.unserved_load

Note: the hawaii version defines an unserved reserves penalty but doesn't apply
the unserved reserves to help meet the reserve requirement, so we drop it here.
Instead, reserves must always be met, and load will be shed to keep reserves
available. This is more like standard practice and makes resource adequacy
evaluation simpler.
"""

import os
from pyomo.environ import *
from switch_model.reporting import write_table


def define_arguments(argparser):
    argparser.add_argument(
        "--unserved-load-penalty",
        type=float,
        default=None,
        help="Penalty to charge per MWh of unserved load. Usually set high enough to force unserved load to zero (default is $10,000/MWh).",
    )


def define_components(m):
    # create an unserved load variable with a high penalty cost,
    # to avoid infeasibilities when
    # evaluating scenarios that are on the edge of infeasibility
    # cost per MWh for unserved load (high)
    if m.options.unserved_load_penalty is not None:
        # always use penalty factor supplied on the command line, if any
        m.unserved_load_penalty_per_mwh = Param(
            within=NonNegativeReals, initialize=m.options.unserved_load_penalty
        )
    else:
        # no penalty on the command line, use whatever is in the parameter files, or 10000
        m.unserved_load_penalty_per_mwh = Param(within=NonNegativeReals, default=10000)

    # amount of unserved load during each timepoint
    m.UnservedLoad = Var(m.LOAD_ZONES, m.TIMEPOINTS, within=NonNegativeReals)
    # total cost for unserved load
    m.UnservedLoadPenalty = Expression(
        m.TIMEPOINTS,
        rule=lambda m, tp: m.tp_duration_hrs[tp]
        * sum(
            m.UnservedLoad[z, tp] * m.unserved_load_penalty_per_mwh
            for z in m.LOAD_ZONES
        ),
    )
    # add the unserved load to the model's energy balance
    m.Zone_Power_Injections.append("UnservedLoad")
    # add the unserved load penalty to the model's objective function
    m.Cost_Components_Per_TP.append("UnservedLoadPenalty")


def post_solve(m, outdir):
    # save unserved load values, skipping timepoints with none
    if hasattr(m, "UnservedLoad"):
        Var(m.LOAD_ZONES, m.TIMEPOINTS, within=NonNegativeReals)
        unserved_mw = {
            (z, tp): m.UnservedLoad[z, tp]
            for z in m.LOAD_ZONES
            for tp in m.TIMEPOINTS
            if abs(value(m.UnservedLoad[z, tp])) > 1e-9
        }
        write_table(
            m,
            unserved_mw.keys(),
            output_file=os.path.join(outdir, "unserved_load.csv"),
            headings=(
                "LOAD_ZONE",
                "TIMEPOINT",
                "UnservedLoadMW",
                "UnservedLoad_GWh_typical_year",
            ),
            values=lambda m, z, tp: (
                z,
                tp,
                unserved_mw[z, tp],
                unserved_mw[z, tp] * m.tp_weight_in_year[tp] * 0.001,
            ),
            digits=16,
        )
