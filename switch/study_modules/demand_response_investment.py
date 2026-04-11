# Copyright (c) 2015-2024 The Switch Authors. All rights reserved.
# Licensed under the Apache License, Version 2, which is in the LICENSE file.

"""
Defines a simple Demand Response Shift Service for the Switch model.
Load in a certain load zone may be shifted between timepoints belonging to the
same date at no cost, which allows assessing the potential value of
demand shifting. This does not include a Shed Service (curtailment of load),
nor a Shimmy Service (fast dispatch for load following or regulation).

This version is identical to switch_model.balancing.demand_response.simple,
except that it adds an optional annual cost to deploy the DR service.
"""

import os
from pyomo.environ import *

dependencies = "switch_model.timescales", "switch_model.balancing.load_zones"
optional_dependencies = "switch_model.transmission.local_td"


def define_components(m):
    """
    Adds components to a Pyomo abstract model object to describe a demand
    response shift service.

    dr_shift_down_limit[(z,t in ZONE_TIMEPOINTS)] is a parameter
    that describes the maximum reduction in demand for load-shifting demand
    response (in MW) that is allowed in a load zone at a specific timepoint.
    Its default value is 0, and it may not exceed the load.

    dr_shift_up_limit[z,t] is a parameter that describes the maximum
    increase in demand for load-shifting demand response (in MW) that is
    allowed in a load zone at a specific timepoint. Its default value is
    infinity.

    ShiftDemand[z,t] is a decision variable describing how much load
    in MW is reduced (if its value is negative) or increased (if
    its value is positive). This variable is bounded by dr_shift_down_limit
    and dr_shift_up_limit.

    If the local_td module is included, ShiftDemand[z,t] will be registered
    with local_td's distributed node for energy balancing purposes. If
    local_td is not included, it will be registered with load zone's central
    node and will not reflect efficiency losses in the distribution network.

    DR_Shift_Net_Zero[z, d in DATES] is a constraint that forces all the
    changes in the demand to balance out over the course of each date.

    TODO: add description of cost components

    When using multi-day timeseries (e.g., weeks or years), you should provide
    tp_date values in timepoints.csv to identify which timepoints fall on the
    same date. Otherwise, load will be shifted freely to any other part of the
    same timeseries (possibly months away).
    """

    m.dr_shift_down_limit = Param(
        m.LOAD_ZONES,
        m.TIMEPOINTS,
        default=0.0,
        within=NonNegativeReals,
        validate=lambda m, value, z, t: value <= m.zone_demand_mw[z, t],
    )
    m.dr_shift_up_limit = Param(
        m.LOAD_ZONES, m.TIMEPOINTS, default=float("inf"), within=NonNegativeReals
    )
    m.dr_annual_cost = Param(m.LOAD_ZONES, m.PERIODS, within=NonNegativeReals)

    m.ShiftDemand = Var(m.LOAD_ZONES, m.TIMEPOINTS, within=Reals)
    m.DeployDRShare = Var(m.LOAD_ZONES, m.PERIODS, within=PercentFraction)

    m.DR_Shift_Lower_Limit = Constraint(
        m.LOAD_ZONES,
        m.TIMEPOINTS,
        rule=lambda m, z, t: m.ShiftDemand[z, t]
        >= -1 * m.DeployDRShare[z, m.tp_period[t]] * m.dr_shift_down_limit[z, t],
    )
    m.DR_Shift_Upper_Limit = Constraint(
        m.LOAD_ZONES,
        m.TIMEPOINTS,
        rule=lambda m, z, t: m.ShiftDemand[z, t]
        <= m.DeployDRShare[z, m.tp_period[t]] * m.dr_shift_up_limit[z, t],
    )

    m.DR_Shift_Net_Zero = Constraint(
        m.LOAD_ZONES,
        m.DATES,
        rule=lambda m, z, d: sum(m.ShiftDemand[z, t] for t in m.TPS_IN_DATE[d]) == 0.0,
    )

    m.DRAnnualCost = Expression(
        m.PERIODS,
        rule=lambda m, p: sum(
            m.DeployDRShare[z, p] * m.dr_annual_cost[z, p] for z in m.LOAD_ZONES
        ),
    )

    try:
        m.Distributed_Power_Withdrawals.append("ShiftDemand")
    except AttributeError:
        m.Zone_Power_Withdrawals.append("ShiftDemand")

    m.Cost_Components_Per_Period.append("DRAnnualCost")


def load_inputs(m, switch_data, inputs_dir):
    """

    Import demand response-specific data from an input directory.

    dr_data.csv
        LOAD_ZONE, TIMEPOINT, dr_shift_down_limit, dr_shift_up_limit

    dr_annual_cost.csv
        LOAD_ZONE, PERIOD, dr_annual_cost
    """

    switch_data.load_aug(
        optional=True,
        filename=os.path.join(inputs_dir, "dr_data.csv"),
        param=(m.dr_shift_down_limit, m.dr_shift_up_limit),
    )
    switch_data.load_aug(
        optional=True,
        filename=os.path.join(inputs_dir, "dr_annual_cost.csv"),
        param=(m.dr_annual_cost,),
    )
