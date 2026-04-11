# Copyright (c) 2015-2024 The Switch Authors. All rights reserved.
# Licensed under the Apache License, Version 2, which is in the LICENSE file.

"""
Allows load to be reduced by a specified amount each timepoint,
in exchange for an annual payment.

TODO: make a better treatment for multi-period models, e.g., capital investment
up front that gives a certain shaped load reduction for a certain number of
years afterward (possibly multiple opportunities per zone/period)
"""

import os
from pyomo.environ import *


def define_components(m):
    """
    Adds components to a Pyomo abstract model object to describe a demand
    response shift service.

    ee_load_reduction[(z, t in ZONE_TIMEPOINTS)] is a parameter that describes
    the reduction in demand that will occur if the full energy-efficiency
    opportunity is deployed. This must be less than zone_demand_mw.

    ee_annual_cost[z, p in LOAD_ZONES * PERIODS] is a parameter that describes
    the annual cost if the full energy-efficiency opportunity is deployed

    DeployEEShare[z, p] is a decision variable setting what fraction of the full
    energy-efficiency opportunity will be deployed

    EEDemandReduction[z, t] is an expression  decision variable describing how much load
    in MW is reduced (if its value is negative) or increased (if
    its value is positive).

    If the local_td module is included, EEDemandReduction[z,t] will be
    registered with local_td's distributed node for energy balancing purposes.
    If local_td is not included, it will be registered with load zone's central
    node and will not reflect efficiency losses in the distribution network.

    TODO: add description of cost components

    """

    m.ee_load_reduction = Param(
        m.LOAD_ZONES,
        m.TIMEPOINTS,
        default=0.0,
        within=NonNegativeReals,
        validate=lambda m, value, z, t: value <= m.zone_demand_mw[z, t],
    )
    m.ee_annual_cost = Param(m.LOAD_ZONES, m.PERIODS, within=NonNegativeReals)
    m.DeployEEShare = Var(m.LOAD_ZONES, m.PERIODS, within=PercentFraction)

    m.EEDemandReduction = Expression(
        m.LOAD_ZONES,
        m.TIMEPOINTS,
        rule=lambda m, z, t: -1
        * m.DeployEEShare[z, m.tp_period[t]]
        * m.ee_load_reduction[z, t],
    )

    m.EEAnnualCost = Expression(
        m.PERIODS,
        rule=lambda m, p: sum(
            m.DeployEEShare[z, p] * m.ee_annual_cost[z, p] for z in m.LOAD_ZONES
        ),
    )

    try:
        m.Distributed_Power_Withdrawals.append("EEDemandReduction")
    except AttributeError:
        m.Zone_Power_Withdrawals.append("EEDemandReduction")

    m.Cost_Components_Per_Period.append("EEAnnualCost")


def load_inputs(m, switch_data, inputs_dir):
    """

    Import demand response-specific data from an input directory.

    ee_data.csv
        LOAD_ZONE, TIMEPOINT, ee_load_reduction

    ee_annual_cost.csv
        LOAD_ZONE, PERIOD, ee_annual_cost
    """

    switch_data.load_aug(
        optional=True,
        filename=os.path.join(inputs_dir, "ee_data.csv"),
        param=(m.ee_load_reduction,),
    )
    switch_data.load_aug(
        optional=True,
        filename=os.path.join(inputs_dir, "ee_annual_cost.csv"),
        param=(m.ee_annual_cost,),
    )
