# Copyright (c) 2015-2024 The Switch Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0, which is in the LICENSE file.
"""
This module defines planning reserves margins to support resource adequacy
requirements. These requirements are sometimes called capacity reserve margins.

Planning reserve margins have been an industry standard for decades that are
roughly defined as: (GenerationCapacity - Demand) / Demand. The idea was that
if you have 15% generation capacity above and beyond demand, the grid could
maintain high reliability. Generation capacity typically includes local
capacity and scheduled imports, while demand typically accounts for demand
response and other distributed energy resources.

This simple definition is problematic for energy-constrained resources such as
hydro, wind, solar, or storage. It also fails to account whether a resource
will be available when it is needed. As this problem became more recognized,
people shifted terminology from "planning reserve margin" to "resource
adequacy requirements" which had more dynamic rules based on time of day,
weather conditions, season, etc.

The "correct" treatment of energy constrained resources is still being debated.
This module implements a simple and flexible treatment, where the user can
specify capacity_value timeseries for any generator, so the available capacity
will be: GenCapacity[g] * capacity_value[g,t]. For renewable resources, this
capacity value timeseries will default to their capacity factor timeseries.

By default, storage and transmission will be credited with their expected
net power delivery.

By default, capacity requirements are equal to standard loads (zone_demand_mw)
plus any local T&D losses, plus the reserve margin.

Other modules can increase the planning reserve requirements by adding
components to the m.REQUIREMENTS_FOR_CAPACITY_RESERVES component list. They can
credit additional capacity toward these requirements by adding modules to the
m.CAPACITY_FOR_RESERVES component list. Components in either of these lists
should be indexed by m.PRR_TIMEPOINTS.

References:

North American Electric Reliability Corporation brief definition and
discussion of planning reserve margins.
http://www.nerc.com/pa/RAPA/ri/Pages/PlanningReserveMargin.aspx

California Independent System Operator Issue paper on Resource Adequacy which
includes both capacity and flexibility requirements. Capacity reserve
requirements can be both system-wide and local, and can potentially accomodate
anything that injects, withdraws or reshapes power. Note that the flexibility
requirements finally includes an energy component, not just ramping capabilities.
http://www.caiso.com/Documents/IssuePaper-RegionalResourceAdequacy.pdf

CA ISO comments filed with the Public Utilities Commission on resource adequacy
rules (and the need to improve them)
https://www.caiso.com/Documents/Jan29_2016_Comments_2017Track1Proposals_ResourceAdequacyProgram_R14-10-010.pdf

"""

import os
from pyomo.environ import *
from switch_model.utilities import unique_list

dependencies = (
    "switch_model.timescales",
    "switch_model.financials",
    "switch_model.balancing.load_zones",
    "switch_model.energy_sources.properties",
    "switch_model.generators.core.build",
    "switch_model.generators.core.dispatch",
)
optional_prerequisites = (
    "switch_model.generators.storage",
    "switch_model.transmission.local_td",
    "switch_model.transmission.transport.build",
    "switch_model.transmission.transport.dispatch",
)


def define_arguments(argparser):
    argparser.add_argument(
        "--find-current-prr",
        action="store_true",
        default=False,
        help="Instead of minimizing costs, find the highest planning reserve margin that can be applied across all regions.",
    )


def define_dynamic_lists(model):
    """
    CAPACITY_FOR_RESERVES is a list of model components than can contribute
    to satisfying planning reserve requirements.

    REQUIREMENTS_FOR_CAPACITY_RESERVES is a corresponding list of model
    components that contribute to planning reserve requirements.

    All components of each list should be indexed by planning reserve
    requirement and timepoint, and be specified in units of MW.
    """
    model.CAPACITY_FOR_RESERVES = []
    model.REQUIREMENTS_FOR_CAPACITY_RESERVES = []


def define_components(m):
    """
    PLANNING_RESERVE_REQUIREMENTS is the set of planning reserve requirements.
    Each planning reserve requirement specifies a certain capacity reserve
    margin be enforced over a certain geographic area in either peak load
    conditions or in every timepoint. Where specified, planning reserve
    requirements are enforced in every investment period. The planning reserve
    area is specified as set of load zones. Typical use cases include
    specifying one planning reserve requirement per load zone, one aggregate
    requirement for the entire system, or a combination of a system-wide
    requirement and requirements for transmission-constrained "load pockets".
    This set is abbreviated as PRR / prr.

    prr_reserve_margin[prr] is the capacity reserve margin for each PRR which
    defaults to 0.15

    prr_enforcement_timescale[prr] Determines whether planning reserve
    requirements are enforced in each timepoint, or just timepoints with peak
    load (zone_demand_mw). Allowed values are 'all_timepoints' and 'peak_load'.

    PRR_ZONES is a set of (prr, zone) that describes which zones contribute to a
    given planning reserve requirement. Zones may belong to more than one PRR.

    PRR_TIMEPOINTS is a sparse set of (prr, t)

    gen_capacity_value[g, t] is a ratio of how much of a generator's installed
    capacity should be credited towards capacity reserve requirements. This
    defaults to gen_max_capacity_factor for renewable projects with variable
    output and 1.0 for other plants.

    AvailableReserveCapacity[prr,t] summarizes the available generation
    capacity across each planning reserve area, taking into account
    capacity_value. If storage projects are being modeled, they are credited
    with their scheduled net deliveries (dispatch - charging). This is added
    to the CAPACITY_FOR_RESERVES list.

    If TXPowerNet is defined in the model, it will be added to the
    CAPACITY_FOR_RESERVES list.

    CapacityRequirements[z,t] is an expression that defines capacity reserve
    requirements. This is set to (1+prr_reserve_margin) * load * local_t&d_loss_factor
    If the local_td module has been included, load will be set to
    WithdrawFromCentralGrid, which accounts for Distributed Energy Resources
    reducing (or increasing) net load to the central grid.
    If the local_td module is not include, load is set to zone_demand_mw and
    will not reflect any DER activities.
    """
    m.PLANNING_RESERVE_REQUIREMENTS = Set(
        dimen=1, doc="Areas and times where planning reserve margins are specified."
    )
    m.PRR_ZONES = Set(
        dimen=2,
        doc=(
            "A set of (prr, z) that describes which zones contribute to each "
            "Planning Reserve Requirement."
        ),
    )
    m.ZONES_IN_PRR = Set(
        m.PLANNING_RESERVE_REQUIREMENTS,
        initialize=lambda m, prr: unique_list(
            z for (_prr, z) in m.PRR_ZONES if _prr == prr
        ),
    )

    m.prr_cap_reserve_margin = Param(
        m.PLANNING_RESERVE_REQUIREMENTS, within=Reals, default=0.15
    )
    m.prr_enforcement_timescale = Param(
        m.PLANNING_RESERVE_REQUIREMENTS,
        default="peak_load",
        within=Any,
        validate=lambda m, value, prr: value in {"all_timepoints", "peak_load"},
        doc=(
            "Determines whether planning reserve requirements are enforced in "
            "each timepoint, or just timepoints with peak load (zone_demand_mw)."
        ),
    )

    def get_peak_timepoints(m, prr):
        """
        Return the set of timepoints with peak load within a planning reserve
        requirement area for each period. For this calculation, load is defined
        statically (zone_demand_mw), ignoring the impact of all distributed
        energy resources.
        """
        peak_timepoint_list = []
        ZONES = [z for (_prr, z) in m.PRR_ZONES if _prr == prr]
        for p in m.PERIODS:
            peak_load = 0.0
            for t in m.TPS_IN_PERIOD[p]:
                load = sum(m.zone_demand_mw[z, t] for z in ZONES)
                if load >= peak_load:
                    peak_timepoint = t
                    peak_load = load
            peak_timepoint_list.append(peak_timepoint)
        return peak_timepoint_list

    def PRR_TIMEPOINTS_init(m):
        PRR_TIMEPOINTS = []
        for prr in m.PLANNING_RESERVE_REQUIREMENTS:
            if m.prr_enforcement_timescale[prr] == "all_timepoints":
                PRR_TIMEPOINTS.extend([(prr, t) for t in m.TIMEPOINTS])
            elif m.prr_enforcement_timescale[prr] == "peak_load":
                PRR_TIMEPOINTS.extend([(prr, t) for t in get_peak_timepoints(m, prr)])
            else:
                raise ValueError(
                    "prr_enforcement_timescale not recognized: '{}'".format(
                        m.prr_enforcement_timescale[prr]
                    )
                )
        return PRR_TIMEPOINTS

    m.PRR_TIMEPOINTS = Set(
        dimen=2,
        within=m.PLANNING_RESERVE_REQUIREMENTS * m.TIMEPOINTS,
        initialize=PRR_TIMEPOINTS_init,
        doc=(
            "The sparse set of (prr, t) for which planning reserve "
            "requirements are enforced."
        ),
    )

    # term used to account for losses between main node and load
    # or avoided losses by using distributed generation
    m.prr_local_td_multiplier = Param(
        m.LOAD_ZONES,
        rule=lambda m, z: (
            1 / (1 - m.local_td_loss_rate[z]) if hasattr(m, "local_td_loss_rate") else 1
        ),
    )

    m.gen_can_provide_cap_reserves = Param(
        m.GENERATION_PROJECTS,
        within=Boolean,
        default=True,
        doc="Indicates whether a generator can provide capacity reserves.",
    )

    def gen_capacity_value_default(m, g, t):
        if not m.gen_can_provide_cap_reserves[g]:
            return 0.0
        elif g in m.VARIABLE_GENS:
            # This can be > 1 (Ex solar on partly cloudy days). Take a
            # conservative approach of capping at 100% of nameplate capacity.
            return min(1.0, m.gen_max_capacity_factor[g, t])
        else:
            return 1.0

    m.gen_capacity_value = Param(
        m.GEN_TPS,
        within=NonNegativeReals,
        default=gen_capacity_value_default,
        validate=lambda m, value, g, t: (
            value == 0.0 if not m.gen_can_provide_cap_reserves[g] else True
        ),
    )

    def AvailableReserveCapacity_rule(m, prr, t):
        reserve_cap = 0.0
        ZONES = m.ZONES_IN_PRR[prr]
        GENS = [
            g
            for z in ZONES
            for g in m.GENS_IN_ZONE[z]
            if (g, t) in m.GEN_TPS and m.gen_can_provide_cap_reserves[g]
        ]
        STORAGE_GENS = getattr(m, "STORAGE_GENS", set())
        for g in GENS:
            if g in STORAGE_GENS:
                # Storage is credited with its scheduled output
                credit = m.DispatchGen[g, t] - m.ChargeStorage[g, t]
            else:
                # Standard generator
                credit = m.gen_capacity_value[g, t] * m.GenCapacityInTP[g, t]

            if m.gen_is_distributed:
                # Give distributed generation credit for avoiding local T&D losses
                # (we ignore the rare case where distributed generation exceeds
                # local loads and therefore has double losses to serve loads in
                # other zones)
                credit *= m.prr_local_td_multiplier[m.gen_load_zone[g]]

            reserve_cap += credit

        return reserve_cap

    m.AvailableReserveCapacity = Expression(
        m.PRR_TIMEPOINTS, rule=AvailableReserveCapacity_rule
    )
    m.CAPACITY_FOR_RESERVES.append("AvailableReserveCapacity")

    if hasattr(m, "TXPowerNet"):
        m.AvailableTXReserveCapacity = Expression(
            m.PRR_TIMEPOINTS,
            rule=lambda m, prr, t: sum(m.TXPowerNet[z, t] for z in m.ZONES_IN_PRR[prr]),
        )
        m.CAPACITY_FOR_RESERVES.append("AvailableTXReserveCapacity")

    if m.options.find_current_prr:
        # maximize a single PRR across all zones (to find the maximum feasible
        # PRR under current conditions)
        m.MaxPRMargin = Var(within=Reals)  # will become model objective in pre_solve()
        # force big to test
        m.ForceMaxPRMargin = Constraint(rule=lambda m: m.MaxPRMargin >= 0.005)
        m.CurrentPRRMargin = Var(m.PLANNING_RESERVE_REQUIREMENTS, within=Reals)
        m.Use_Same_PRR_Across_Regions = Constraint(
            m.PLANNING_RESERVE_REQUIREMENTS,
            rule=lambda m, prr: m.CurrentPRRMargin[prr] == m.MaxPRMargin,
        )

    def CapacityRequirements_rule(m, prr, t):
        # capacity requirements are defined at the zone's central node, so we increase
        # loads to account for local T&D losses, if any
        prr_var = (
            "CurrentPRRMargin"
            if m.options.find_current_prr
            else "prr_cap_reserve_margin"
        )
        return sum(
            (1 + getattr(m, prr_var)[prr])
            * m.zone_demand_mw[z, t]
            * m.prr_local_td_multiplier[z]
            for z in m.ZONES_IN_PRR[prr]
        )

    m.CapacityRequirements = Expression(
        m.PRR_TIMEPOINTS, rule=CapacityRequirements_rule
    )
    m.REQUIREMENTS_FOR_CAPACITY_RESERVES.append("CapacityRequirements")


def define_dynamic_components(model):
    """ """
    model.Enforce_Planning_Reserve_Margin = Constraint(
        model.PRR_TIMEPOINTS,
        rule=lambda m, prr, t: (
            sum(
                getattr(m, reserve_cap)[prr, t]
                for reserve_cap in m.CAPACITY_FOR_RESERVES
            )
            >= sum(
                getattr(m, cap_requirement)[prr, t]
                for cap_requirement in m.REQUIREMENTS_FOR_CAPACITY_RESERVES
            )
        ),
        doc=(
            "Ensures that the sum of CAPACITY_FOR_RESERVES satisfies the sum "
            "of REQUIREMENTS_FOR_CAPACITY_RESERVES for each of PRR_TIMEPOINTS."
        ),
    )


def pre_solve(m):
    if m.options.find_current_prr:
        # deactivate current objective and maximize MaxPRMargin instead
        for o in m.component_objects(Objective, active=True):
            o.deactivate()
        m.Maximize_PRMargin = Objective(rule=lambda m: m.MaxPRMargin, sense=maximize)


def post_solve(m, outputs_dir):
    if m.options.find_current_prr:
        m.logger.info(
            "\nSolved for highest attainable planning reserve margin shared across all regions:"
            f"\n{value(m.MaxPRMargin)}"
            "\nNOTE: the `--find-current-prr` flag was used for this run. This maximizes the "
            "\nplanning reserve margin, ignoring financial costs. Results from this run (other "
            "\nthan the reserve margin calculation) should not be used for analysis.\n"
        )


def load_inputs(model, switch_data, inputs_dir):
    """
    Files or columns marked with * are optional. See notes above on default
    values.

    reserve_capacity_value.csv*
        GENERATION_PROJECT, TIMEPOINT, gen_capacity_value

    planning_reserve_requirements.csv*
        PLANNING_RESERVE_REQUIREMENT, prr_cap_reserve_margin*, prr_enforcement_timescale*

    gen_info.csv
        ..., gen_can_provide_cap_reserves*

    planning_reserve_requirement_zones.csv
        PLANNING_RESERVE_REQUIREMENT, LOAD_ZONE

    """
    switch_data.load_aug(
        filename=os.path.join(inputs_dir, "reserve_capacity_value.csv"),
        optional=True,
        param=(model.gen_capacity_value),
    )
    switch_data.load_aug(
        filename=os.path.join(inputs_dir, "planning_reserve_requirements.csv"),
        optional=True,
        index=model.PLANNING_RESERVE_REQUIREMENTS,
        param=(model.prr_cap_reserve_margin, model.prr_enforcement_timescale),
    )
    switch_data.load_aug(
        filename=os.path.join(inputs_dir, "gen_info.csv"),
        param=(model.gen_can_provide_cap_reserves),
    )
    switch_data.load_aug(
        filename=os.path.join(inputs_dir, "planning_reserve_requirement_zones.csv"),
        set=model.PRR_ZONES,
    )
