"""
Set ramp-rate limits for generators.

Capacity that is already online can ramp by gen_ramp_limit_up or
gen_ramp_limit_down per hour. These should be expressed as a decimal fraction of
the committed capacity (0-1).

For multi-hour timepoints, capacity starting up for each timepoint can jump at
least to gen_min_load_fraction in the first hour, then ramp up from there at
gen_ramp_limit_up for the rest of the timepoint. For timepoints of one hour or
less, capacity starting up can ramp by at least gen_min_load_fraction over the
course of the timepoint. These rules will work consistently when using
single-hour or multi-hour timepoints, but will allow faster startup ramps when
using timepoints shorter than one hour. Similar rules apply for capacity
shutting down during each timepoint: it can ramp down by the normal rate
(gen_ramp_limit_down) until the last hour, of the timepoint, then all the way
from gen_min_load_fraction down to zero in the last hour, and always at least
gen_min_load_fraction per timepoint.

To reduce memory use, no constraint is defined if the generator could jump all
the way from zero to full load (or full load to zero) in one timepoint following
the rules above.

Calculations in this module are based on an assumption that CommitGen and
DispatchGen are instantaneous values at the start of each timepoint. (Sometimes
average loads for the hour starting at that point or center-hour availability
for that hour are used for these "instantaneous" values, since those are all
that is available.) Startups, shutdowns and ramps begin at the start of the
_prior_ timepoint (just after CommitGen and DispatchGen are set for that
timepoint) and ramp until the the start of _this_ timepoint, when new CommitGen
and DispatchGen values occur. So ramp limits are calculated based on the amount
of time from the prior timepoint to the current timepoint and the amount of
capacity started up or shutdown for the current timepoint.
"""

import os
from pyomo.environ import *


def define_components(m):
    """ """
    # maximum fraction of committed capacity that can be ramped up or down per
    # hour
    m.gen_ramp_limit_up = Param(
        m.GENERATION_PROJECTS, within=NonNegativeReals, default=1
    )
    m.gen_ramp_limit_down = Param(
        m.GENERATION_PROJECTS, within=NonNegativeReals, default=1
    )

    def ramp_up_rule(m, g, tp):
        # amount of time available for ramping from previous timepoint to this
        # one
        tp_dur = m.tp_duration_hrs[m.tp_previous[tp]]
        # ramp occurs during previous timepoint, but we use min-load fraction
        # for the current timepoint to ensure startup can reach this level
        min_load_frac = m.gen_min_load_fraction_TP[g, tp]
        ramp_up_limit = m.gen_ramp_limit_up[g]
        # Maximum possible ramp in a timepoint, including a startup. See module
        # notes for this calculation.
        startup_max_ramp = max(
            min_load_frac,
            max(min_load_frac, ramp_up_limit) + ramp_up_limit * (tp_dur - 1),
        )
        if startup_max_ramp >= 1:
            # gen can always ramp to max output (100%) within one timepoint, so
            # there is no need to enforce ramp rates. This test is based on the
            # following:
            # (1) if capacity is started up for this timepoint, it can ramp to
            # full power (by definition)
            # (2) if capacity is already running, then it must already be at min
            # load, so it has a 1-hour headstart vs. being started up, which
            # means it can also reach full power by the end of the timepoint
            return Constraint.Skip

        # max ramp calculation is similar to GenX, but handles multi-hour timepoints
        # https://github.com/GenXProject/GenX.jl/blob/62c0c0c/src/model/resources/thermal/thermal_commit.jl#L187
        max_ramp_up = (
            # capacity that was previously on and is still on now (wasn't just
            # started up) respects the ramp limits
            ramp_up_limit * (m.CommitGen[g, tp] - m.StartupGenCapacity[g, tp]) * tp_dur
            # capacity started up this timepoint follows the startup_max_ramp
            # calculation
            + m.StartupGenCapacity[g, tp] * startup_max_ramp
            # capacity stopping this timepoint must withdraw at least min load
            - m.ShutdownGenCapacity[g, tp] * min_load_frac
        )
        ramp_up = m.DispatchGen[g, tp] - m.DispatchGen[g, m.tp_previous[tp]]
        return ramp_up <= max_ramp_up

    m.Max_Ramp_Up = Constraint(m.GEN_TPS, rule=ramp_up_rule)

    def ramp_down_rule(m, g, tp):
        tp_dur = m.tp_duration_hrs[m.tp_previous[tp]]
        min_load_frac = m.gen_min_load_fraction_TP[g, tp]
        ramp_down_limit = m.gen_ramp_limit_down[g]
        # capacity stopping for this timepoint can ramp down normally in the
        # time before the final hour, then ramp down by at least min load in the
        # last hour (or at least min-load for fractional-hour timepoints).
        shutdown_max_ramp = max(
            min_load_frac,
            max(min_load_frac, ramp_down_limit) + ramp_down_limit * (tp_dur - 1),
        )
        if shutdown_max_ramp >= 1:
            # gen can ramp >= 100% per timepoint
            return Constraint.Skip
        max_ramp_down = (
            # capacity that was previously on and is still on now (wasn't just
            # started up) respects the ramp limits
            ramp_down_limit
            * (m.CommitGen[g, tp] - m.StartupGenCapacity[g, tp])
            * tp_dur
            # capacity shutting down this timepoint follows the
            # shutdown_max_ramp calculation
            + m.ShutdownGenCapacity[g, tp] * shutdown_max_ramp
            # capacity starting up this timepoint must at least jump to min load
            - m.StartupGenCapacity[g, tp] * min_load_frac
        )
        ramp_down = m.DispatchGen[g, m.tp_previous[tp]] - m.DispatchGen[g, tp]
        return ramp_down <= max_ramp_down

    m.Max_Ramp_Down = Constraint(m.GEN_TPS, rule=ramp_down_rule)


def load_inputs(m, switch_data, inputs_dir):
    """
    Import O&M parameters. All columns are optional.

    gen_info.csv
        GENERATION_PROJECT,
        gen_ramp_limit_up,
        gen_ramp_limit_down,
    """

    switch_data.load_aug(
        filename=os.path.join(inputs_dir, "gen_info.csv"),
        optional=True,
        param=(
            m.gen_ramp_limit_up,
            m.gen_ramp_limit_down,
        ),
    )
