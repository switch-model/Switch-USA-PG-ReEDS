"""
This adds an enhanced treatment of scheduled outages: require the  model to
schedule fractions of the capacity to be offline for full sample days, with the
total unavailability adding up to match the scheduled outage rate. This will
avoid cherry-picking the individual peak hours. With this framework, Switch
would probably schedule capacity to be offline on the super-peak PRM day (since
that has 0 probability weight) plus the peakiest of the other days. This may be
a little over-optimistic, depending how predictable we think super-peak
conditions are (probably fairly predictable if based on weather, but less
predictable if based on coincident plant outages).

TODO: maybe change GenCapacityInTP to be GenCapacity - ScheduledOutage?
(Maybe also minus ForcedOutage?) That could reduce duplication of rules and clarify
what is going on, e.g., for PRM, etc.
"""

from pyomo.environ import *


def define_components(m):
    # note: for baseload gens, scheduled outage is prorated year-round
    # so here we are only interested in non-baseload gens
    m.SCHEDULED_OUTAGE_GEN_PERIODS = Set(
        dimen=2,
        within=m.GEN_PERIODS,
        initialize=lambda m: [
            (g, p)
            for g in m.GENERATION_PROJECTS
            for p in m.PERIODS_FOR_GEN[g]
            if m.gen_scheduled_outage_rate[g] > 0 and m.gen_is_baseload[g] == 0
        ],
    )
    m.SCHEDULED_OUTAGE_GEN_TS = Set(
        dimen=2,
        within=m.GENERATION_PROJECTS * m.TIMESERIES,
        initialize=lambda m: [
            (g, ts)
            for g, p in m.SCHEDULED_OUTAGE_GEN_PERIODS
            for ts in m.TS_IN_PERIOD[p]
        ],
    )
    m.SCHEDULED_OUTAGE_GEN_TPS = Set(
        dimen=2,
        within=m.GEN_TPS,
        initialize=lambda m: [
            (g, tp) for g, ts in m.SCHEDULED_OUTAGE_GEN_TS for tp in m.TPS_IN_TS[ts]
        ],
    )

    # MW scheduled offline each timeseries
    m.ScheduleOutage = Var(m.SCHEDULED_OUTAGE_GEN_TS, within=NonNegativeReals)

    # make sure the scheduled outage rate is met for each SO gen each period
    m.ts_hours_per_year = Param(
        m.TIMESERIES, rule=lambda m, ts: m.ts_duration_hrs[ts] * m.ts_scale_to_year[ts]
    )
    # timeseries in each period may not add up to exactly 8760 hours, so we just
    # go with whatever they do add up to
    m.hours_per_year = Param(
        m.PERIODS,
        rule=lambda m, p: sum(m.ts_hours_per_year[ts] for ts in m.TS_IN_PERIOD[p]),
    )
    m.Schedule_Enough_Outages = Constraint(
        m.SCHEDULED_OUTAGE_GEN_PERIODS,
        rule=lambda m, g, p: sum(
            m.ScheduleOutage[g, ts] * m.ts_hours_per_year[ts]
            for ts in m.TS_IN_PERIOD[p]
        )
        >= m.gen_scheduled_outage_rate[g] * m.GenCapacity[g, p] * m.hours_per_year[p],
    )

    if hasattr(m, "CommitGen"):
        # apply a tighter version of CommitUpperLimit; note: we allow capacity
        # forced off by gen_max_commit_fraction to count toward the scheduled
        # outage requirement
        def rule(m, g, tp):
            return (
                m.CommitGen[g, tp]
                <= (m.GenCapacityInTP[g, tp] - m.ScheduleOutage[g, m.tp_ts[tp]])
                * m.gen_availability[g]
            )

    else:
        # apply a tighter version of DispatchUpperLimit (from no_commit module)
        # note: weather-based capacity factor limits don't count toward the
        # scheduled outage requirement
        def rule(m, g, tp):
            cf = m.gen_max_capacity_factor[g, tp] if g in m.VARIABLE_GENS else 1
            return (
                m.DispatchGen[g, tp]
                <= (m.GenCapacityInTP[g, tp] - m.ScheduleOutage[g, m.tp_ts[tp]])
                * m.gen_availability[g]
                * cf
            )

    m.Apply_Scheduled_Outage = Constraint(m.SCHEDULED_OUTAGE_GEN_TPS, rule=rule)
