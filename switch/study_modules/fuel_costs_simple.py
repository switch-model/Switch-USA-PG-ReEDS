# Copyright (c) 2015-2024 The Switch Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0, which is in the LICENSE file.

"""

A simple description of flat fuel costs for the Switch model that
serves as an alternative to the more complex fuel_markets with tiered
supply curves. This is mutually exclusive with the fuel_markets module.

"""
import os
from pyomo.environ import *

dependencies = (
    "switch_model.timescales",
    "switch_model.balancing.load_zones",
    "switch_model.energy_sources.properties.properties",
    "switch_model.generators.core.build",
    "switch_model.generators.core.dispatch",
)


def define_components(mod):
    """
    ZONE_FUEL_PERIODS is a set of (load_zone, fuel, period) for which fuel_cost
    has been provided.

    fuel_cost[(z, f, p) in ZONE_FUEL_PERIODS] describes flat fuel costs for each
    supply of fuel. Costs can vary by load zone and period.

    GEN_TP_FUELS_UNAVAILABLE is a subset of GEN_TP_FUELS that describes which
    points don't have fuel available.

    Enforce_Fuel_Unavailability[(g, t, f) in GEN_TP_FUELS_UNAVAILABLE] is a
    constraint that restricts GenFuelUseRate to 0 in load zones and periods
    where the project's fuel is not available for purchase.

    FuelCostsPerTP[t in TIMEPOINTS] is an expression that summarizes fuel costs
    for the objective function.

    """

    mod.ZONE_FUEL_PERIODS = Set(
        dimen=3,
        validate=lambda m, z, f, p: (
            z in m.LOAD_ZONES and f in m.FUELS and p in m.PERIODS
        ),
    )
    mod.fuel_cost = Param(mod.ZONE_FUEL_PERIODS, within=NonNegativeReals)
    mod.min_data_check("ZONE_FUEL_PERIODS", "fuel_cost")

    mod.GEN_TP_FUELS_UNAVAILABLE = Set(
        dimen=3,
        initialize=mod.GEN_TP_FUELS,
        filter=lambda m, g, t, f: (m.gen_load_zone[g], f, m.tp_period[t])
        not in m.ZONE_FUEL_PERIODS,
    )
    mod.Enforce_Fuel_Unavailability = Constraint(
        mod.GEN_TP_FUELS_UNAVAILABLE,
        rule=lambda m, g, t, f: m.GenFuelUseRate[g, t, f] == 0,
    )

    # Summarize total fuel costs in each timepoint for the objective function
    mod.FuelCostsPerTP = Expression(
        mod.TIMEPOINTS,
        rule=lambda m, tp: sum(
            m.GenFuelUseRate[g, tp, f]
            * m.fuel_cost[m.gen_load_zone[g], f, m.tp_period[tp]]
            for g in m.FUEL_BASED_GENS_IN_PERIOD[m.tp_period[tp]]
            for f in m.FUELS_FOR_GEN[g]
            if (g, tp, f) not in m.GEN_TP_FUELS_UNAVAILABLE
        ),
    )
    mod.Cost_Components_Per_TP.append("FuelCostsPerTP")


def load_inputs(mod, switch_data, inputs_dir):
    """
    Import simple fuel cost data. The following file is expected in
    the input directory:

    fuel_cost.csv
        load_zone, fuel, period, fuel_cost

    """

    switch_data.load_aug(
        filename=os.path.join(inputs_dir, "fuel_cost.csv"),
        index=mod.ZONE_FUEL_PERIODS,
        param=[mod.fuel_cost],
    )
