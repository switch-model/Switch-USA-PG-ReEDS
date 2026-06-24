"""
Report total, coincident, fuel-based production for every timepoint in the
current model.
"""

import os
from pyomo.environ import value
from switch_model.reporting import write_table


def post_solve(m, outdir):
    gens = {
        (grp, z, p): []
        for grp in ["fuel", "variable", "storage", "other"]
        for z in m.LOAD_ZONES
        for p in m.PERIODS
    }
    for g in m.GENERATION_PROJECTS:
        if g in m.FUEL_BASED_GENS:
            group = "fuel"
        elif g in m.VARIABLE_GENS:
            group = "variable"
        elif g in m.STORAGE_GENS:
            group = "storage"
        else:
            group = "other"
        z = m.gen_load_zone[g]
        for p in m.PERIODS_FOR_GEN[g]:
            gens[group, z, p].append(g)

    write_table(
        m,
        m.LOAD_ZONES,
        m.TIMEPOINTS,
        output_file=os.path.join(outdir, "zonal_total_dispatch.csv"),
        headings=(
            "load_zone",
            "period",
            "timeseries",
            "timepoint",
            "fuel_based_gen_dispatch",
            "variable_gen_dispatch",
            "other_gen_dispatch",
        ),
        values=lambda m, z, tp: (
            z,
            m.tp_period[tp],
            m.tp_ts[tp],
            tp,
            sum(m.DispatchGen[g, tp] for g in gens["fuel", z, m.tp_period[tp]]),
            sum(m.DispatchGen[g, tp] for g in gens["variable", z, m.tp_period[tp]]),
            sum(
                m.DispatchGen[g, tp] - m.ChargeStorage[g, tp]
                for g in gens["storage", z, m.tp_period[tp]]
            )
            + sum(m.DispatchGen[g, tp] for g in gens["other", z, m.tp_period[tp]]),
        ),
    )
