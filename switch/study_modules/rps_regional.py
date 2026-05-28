import os, math
from collections import defaultdict
from types import SimpleNamespace
from typing import List, Tuple, Dict

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from pyomo.environ import *

from switch_model.utilities import unique_list

"""
This module models regional RPS/CES compliance by tracking REC production from
eligible generators. Each generator is assigned to an "eligibility group",
identifying all the RPS/CES programs that it is eligible to participate in. Sets
of RPS/CES programs covering a particular geographic region are labeled as
"requirements groups". Generation from each eligibility group may either create
local RECs for programs that cover the generator's zone, or be exported to one
or more requirements groups as bundled RECs (BRECs) or unbundled RECs (URECs).
BRECs are tracked for each timepoint and require matching transmission flows.
URECs are tracked by period without transmission constraints. RECs delivered to
a requirements group may support all eligible RPS/CES programs included in that
group, e.g., a requirements group could consist of both RPS and CES programs in
a paritcular state, and then the same RECs can be used for both simultaneously
if the plant has been marked eligible for both.

Distributed generation and VPP generators (interruptible loads)
(gen_is_distributed or gen_is_vpp) are subtracted from load before the RPS
calculation if not marked as RPS-eligible.

This module does not allow eligibility determination based on fuel used in
multi-fuel generators. See switch_model.hawaii.rps for an example of that.

Transfers go from zone-eligibility group to requirements group where the
requirements groups are all unique subsets of the eligibility group. The module
also manages congestion on routes affected by these flows, assuming transfers
from each zone-eligibility group will travel along the lowest-loss path to the
requirements group.

Note: this module runs best with Pyomo 6.8 or later; earlier versions will give
a lot of warnings about adding duplicate items to sets via Set.add().
"""


def progs_to_group(progs):
    """
    Turn a set of RPS programs into a stable group name. `progs` contains either
    RPS program names or tuples of (RPS program, local flag, bundled flag,
    unbundled flag).
    """
    if progs:
        if not isinstance(progs, list):
            progs = list(progs)
        if isinstance(progs[0], tuple):
            return " / ".join(
                sorted(
                    pr
                    + " "
                    + "+".join(
                        (["local"] if l else [])
                        + (["BREC"] if b else [])
                        + (["UREC"] if u else [])
                    )
                    for pr, l, b, u in progs
                )
            )
        elif isinstance(progs[0], str):
            return " / ".join(sorted(pr for pr in progs))
    return ""


def sorted_tuple(values):
    return tuple(sorted(values))


def sorted_tuple_dict(d):
    return defaultdict(tuple, {key: sorted_tuple(values) for key, values in d.items()})


def make_pyomo_compatible(ns):
    """
    Convert attributes in namespace `ns` into Pyomo-compatible initializers, in
    place. (Pyomo ordered Sets cannot be initialized from raw Python sets.)
    """
    for attr, val in vars(ns).items():
        if isinstance(val, set):
            setattr(ns, attr, sorted_tuple(val))
        elif isinstance(val, defaultdict) and val.default_factory is set:
            setattr(ns, attr, sorted_tuple_dict(val))


def build_rps_policy_topology(m):
    """
    Build all derived RPS/CES topology tables in one pass. Pyomo components are
    initialized from this cache instead of repeating similar scans in multiple
    BuildActions.
    """
    t = SimpleNamespace(
        zone_reqs=defaultdict(set),
        zones_in_rps_program_period=defaultdict(set),
        gens_in_rps_program_period=defaultdict(set),
        rps_gens_in_period=defaultdict(set),
        rps_programs_for_gen_period=defaultdict(set),
        create_local_recs={},
        eligibility_groups=set(),
        rps_programs_for_eligibility_group=defaultdict(set),
        zone_eligibility_group_periods=set(),
        gens_in_zone_eligibility_group_period=defaultdict(set),
        brec_progs_for_eligibility_group=defaultdict(set),
        urec_progs_for_eligibility_group=defaultdict(set),
        local_progs_for_eligibility_group=defaultdict(set),
        requirements_groups=set(),
        requirements_group_periods=set(),
        zones_in_requirements_group_period=defaultdict(set),
        rgs_for_rps_program_period=defaultdict(set),
        brec_routes=set(),
        urec_routes=set(),
        brec_routes_for_zone_eligibility_group_period=defaultdict(set),
        urec_routes_for_zone_eligibility_group_period=defaultdict(set),
        brec_routes_for_rps_program_period=defaultdict(set),
        urec_routes_for_rps_program_period=defaultdict(set),
        local_zone_eligibility_group_periods_for_rps_program_period=defaultdict(set),
    )

    # Program-period requirements define the zones covered by each program and
    # the requirement groups used as REC destinations.
    # (RPS_RULES are the first 3 columns of rps_requirements.csv)
    for pr, pe, z in m.RPS_RULES:
        t.zones_in_rps_program_period[pr, pe].add(z)
        t.zone_reqs[z, pe].add(pr)

    # Generator eligibility rows define eligible gens, reverse indexes, and
    # whether each program-period-gen can create local RECs.
    # (RPS_PROGRAM_PERIOD_GENS are the first 3 columns of rps_generators.csv)
    for pr, pe, g in m.RPS_PROGRAM_PERIOD_GENS:
        t.rps_gens_in_period[pe].add(g)
        t.gens_in_rps_program_period[pr, pe].add(g)
        t.rps_programs_for_gen_period[g, pe].add(pr)
        t.create_local_recs[pr, pe, g] = (
            m.gen_load_zone[g] in t.zones_in_rps_program_period[pr, pe]
        )

    # Eligibility groups aggregate generators with identical REC destinations.
    for pe in m.PERIODS:
        for g in t.rps_gens_in_period[pe]:
            progs = [
                (
                    pr,
                    t.create_local_recs[pr, pe, g],
                    bool(m.send_bundled_recs[pr, pe, g]),
                    bool(m.send_unbundled_recs[pr, pe, g]),
                )
                for pr in t.rps_programs_for_gen_period[g, pe]
            ]
            eg = progs_to_group(progs)
            z = m.gen_load_zone[g]
            t.eligibility_groups.add(eg)
            t.zone_eligibility_group_periods.add((z, eg, pe))
            t.gens_in_zone_eligibility_group_period[z, eg, pe].add(g)

            for pr, local, bundled, unbundled in progs:
                # list of programs served by each eligibility group
                t.rps_programs_for_eligibility_group[eg].add(pr)
                # programs that RECs can be sent to from each eligibility group
                if local:
                    t.local_progs_for_eligibility_group[eg].add(pr)
                if bundled:
                    t.brec_progs_for_eligibility_group[eg].add(pr)
                if unbundled:
                    t.urec_progs_for_eligibility_group[eg].add(pr)

    # Requirements groups aggregate zones with identical program obligations
    for (z, pe), progs in t.zone_reqs.items():
        rg = progs_to_group(progs)
        t.requirements_groups.add(rg)
        t.requirements_group_periods.add((rg, pe))
        t.zones_in_requirements_group_period[rg, pe].add(z)
        # Requirements groups that each RPS program/period participates in
        for pr in progs:
            t.rgs_for_rps_program_period[pr, pe].add(rg)

    # BREC/UREC route tables and inverse indexes for efficient summations
    for z, eg, pe in t.zone_eligibility_group_periods:
        for pr in t.rps_programs_for_eligibility_group[eg]:
            routes = {(z, eg, rg, pe) for rg in t.rgs_for_rps_program_period[pr, pe]}
            if pr in t.brec_progs_for_eligibility_group[eg]:
                t.brec_routes.update(routes)
                t.brec_routes_for_zone_eligibility_group_period[z, eg, pe].update(
                    routes
                )
                t.brec_routes_for_rps_program_period[pr, pe].update(routes)
            if pr in t.urec_progs_for_eligibility_group[eg]:
                t.urec_routes.update(routes)
                t.urec_routes_for_zone_eligibility_group_period[z, eg, pe].update(
                    routes
                )
                t.urec_routes_for_rps_program_period[pr, pe].update(routes)
            if pr in t.local_progs_for_eligibility_group[eg]:
                t.local_zone_eligibility_group_periods_for_rps_program_period[
                    pr, pe
                ].add((z, eg, pe))

    make_pyomo_compatible(t)
    return t


def build_brec_transmission_topology(m):
    """
    Map each policy-valid BREC route onto the transmission network. This
    includes the least-loss path, cumulative delivery efficiency, and inverse
    line/period indexes used by transmission constraints.
    """
    t = SimpleNamespace(
        zones_on_brec_route={},
        brec_route_zones=[],
        brec_route_efficiency_to_zone={},
        brec_routes_using_directional_tx_in_period=defaultdict(set),
        directional_tx_periods_on_any_brec_route=set(),
        brec_tx_period_timepoints=[],
    )

    # Find the lowest-loss route from zone `z` to any zone in the requirements
    # group `rg` in period `pe` for all (z, eg, rg, pe) in m.BREC_ROUTES. Then
    # use that to populate m.ZONES_ON_BREC_ROUTE[z, eg, rg, pe]. This uses
    # Dijkstra's method to find the routes from all zones in rg to all source
    # zones at the same time, to improve efficiency.

    # First, cluster routes with common rg and period, then for each rg/period,
    # find the paths to all the relevant zones, then use that to fill in
    # m.ZONES_ON_BREC_ROUTE[z, eg, rg, pe] for all the relevant routes.
    # We could simplify this by creating EXTERNAL_ZONE_REQUIRMENT_GROUPS and
    # defining ZONES_ON_BREC_ROUTE over that instead of over all BREC_ROUTES,
    # but that might make the overall code a little harder to follow. So instead
    # we (1) find all zones that feed into each rg; (2) find the shortest path
    # (list of zone hops) from each of those zones to the rg; and (3) apply that
    # path to all BREC routes that use that zone and rg.

    # create a list of tuples of "costs" to move power from any zone to a
    # neighbor for the route-finder; this uses the negative log of efficiency so
    # that minimizing the sum of this "cost" across hops minimizes 1/(eff1 *
    # eff2 * ...), i.e., minimizes 1/efficiency, i.e., maximizes efficiency.
    edges = [
        (z1, z2, -math.log(eff))
        for z1, z2 in m.DIRECTIONAL_TX
        for eff in [value(m.trans_efficiency[m.trans_d_line[z1, z2]])]
        if eff > 0
    ]

    # find all zones that export BRECS to each rg in each period
    source_zones_for_rg_period = defaultdict(set)
    for source_z, eg, rg, pe in m.BREC_ROUTES:
        source_zones_for_rg_period[rg, pe].add(source_z)

    # For each rg/period, find the "shortest" path from all zones that export to
    # it
    routes_to_rg_period = {}
    for (rg, pe), source_zones in source_zones_for_rg_period.items():
        routes_to_rg_period[rg, pe] = find_paths(
            from_zones=source_zones,
            to_zones=m.ZONES_IN_REQUIREMENTS_GROUP_PERIOD[rg, pe],
            edges=edges,
        )

    # Apply assembled paths to all matching BREC routes
    for rte in m.BREC_ROUTES:
        source_z, eg, rg, pe = rte
        steps = routes_to_rg_period[rg, pe][source_z]
        if steps is None:
            rg_zones = ", ".join(m.ZONES_IN_REQUIREMENTS_GROUP_PERIOD[rg, pe])
            raise ValueError(
                f"No transmission route could be found from zone '{source_z}' "
                f"to requirements group '{rg}' in period '{pe}' ({rg_zones}). "
                "Either the transmission network is incomplete or generators "
                "in an unconnected zone have been marked eligible for bundled "
                "REC trade, which is not possible."
            )

        t.zones_on_brec_route[rte] = tuple(steps)
        prev_z = None
        prev_efficiency = 1.0
        for z in steps:
            if prev_z is None:
                route_efficiency = 1.0
            else:
                line = m.trans_d_line[prev_z, z]
                route_efficiency = prev_efficiency * value(m.trans_efficiency[line])
                t.brec_routes_using_directional_tx_in_period[prev_z, z, pe].add(rte)
                # transmission corridor/period combos that are affected by BREC trade
                t.directional_tx_periods_on_any_brec_route.add((prev_z, z, pe))

            t.brec_route_zones.append(rte + (z,))
            t.brec_route_efficiency_to_zone[rte + (z,)] = route_efficiency
            prev_z = z
            prev_efficiency = route_efficiency

    for z_from, z_to, pe in t.directional_tx_periods_on_any_brec_route:
        for tp in m.TPS_IN_PERIOD[pe]:
            t.brec_tx_period_timepoints.append((z_from, z_to, pe, tp))

    make_pyomo_compatible(t)
    return t


def find_paths(from_zones, to_zones, edges):
    """
    For each zone in `from_zones`, find the least-cost path to any zone in
    `to_zones` using non-negative directed edge costs [(from_zone, to_zone,
    cost), ...]. Returns a dict {from_zone1: [zone1, zone2, ..., to_zone]}. If a
    source cannot reach any destination, value will be None.
    """
    # TODO: improve documentation, maybe simplify code

    # Build node index map (strings -> ints)
    nodes = set()
    for u, v, _ in edges:
        nodes.add(u)
        nodes.add(v)
    nodes.update(from_zones)
    nodes.update(to_zones)
    nodes = sorted(nodes)
    n = len(nodes)
    index = {z: i for i, z in enumerate(nodes)}

    # Build sparse adjacency (forward) and transpose (reversed)
    rows = np.fromiter((index[u] for u, _, _ in edges), dtype=int, count=len(edges))
    cols = np.fromiter((index[v] for _, v, _ in edges), dtype=int, count=len(edges))
    data = np.fromiter((float(w) for _, _, w in edges), dtype=float, count=len(edges))
    A = csr_matrix((data, (rows, cols)), shape=(n, n))

    AT = A.T  # reversed graph

    # Multi-source Dijkstra from all to_zones on reversed graph dist[i, j] =
    # distance from to_zones[i] to node j (in reversed graph), which equals
    # distance from node j to to_zones[i] in the forward graph.
    sources = [index[z] for z in to_zones]
    dist, pred = dijkstra(AT, directed=True, indices=sources, return_predecessors=True)

    # Convert to 2D if there's a single source
    if dist.ndim == 1:
        dist = dist[np.newaxis, :]
        pred = pred[np.newaxis, :]

    # For each from_zone, pick the best destination and reconstruct path
    def reconstruct_path(u_idx: int, src_row: int) -> List[str]:
        """
        Reconstruct forward path: u -> ... -> m (where m == to_zones[src_row]).
        Uses predecessor row for that specific source.
        """
        path = [u_idx]
        cur = u_idx
        pr = pred[src_row]
        # Walk predecessors until we reach the destination source node
        dest_idx = sources[src_row]
        # If unreachable, pred[cur] == -9999 (SciPy sentinel)
        while cur != dest_idx:
            cur = pr[cur]
            if cur == -9999:
                return []  # unreachable
            path.append(cur)
        return [nodes[i] for i in path]

    results = {}
    for fz in from_zones:
        j = index[fz]
        # distances from each destination to this source (in reversed search)
        col = dist[:, j]
        k = int(np.argmin(col))
        best = col[k]
        if not np.isfinite(best):
            results[fz] = None
            continue
        path = reconstruct_path(j, k)
        if path:
            results[fz] = path
        else:
            results[fz] = None

    return results


def define_components(m: AbstractModel):
    """
    RPS_PROGRAM labels identify program/regions that have target shares of
    clean power (typically a renewable portfolio standard (RPS) or clean energy
    standard (CES)).

    RPS_GEN identifies generators that are eligible for producing clean power
    in at least one RPS/CES program and period.

    LOAD_ZONE identifies model regions where RPS/CES goals are defined.

    rps_share[program, period, zone] is the fraction of net load in a zone and
    period that must be covered by local, bundled or unbundled RECs that are
    eligible for the program.
    """
    # indexing set for the zonal requirements: (program, period, zone) combination
    # (These are all the index columns from rps_requirements.csv.)
    m.RPS_RULES = Set(dimen=3, within=Any * m.PERIODS * m.LOAD_ZONES)

    # share target specified for each (program, period, zone) combination
    m.rps_share = Param(m.RPS_RULES, default=float("inf"), within=Reals)

    # fraction of required RECs in each program/period that can come from
    # unbundled trade
    m.unbundled_rec_limit_fraction = Param(
        m.RPS_RULES, within=PercentFraction, default=1
    )

    if not hasattr(m, "gen_is_vpp"):
        m.gen_is_vpp = Param(m.GENERATION_PROJECTS, within=Binary, default=0)
        m.gen_is_vpp.added_by = __name__

    # names of all the RPS programs and periods when they are in effect;
    # each unique pair of values in the first two columns of
    # rps_requirements.csv is a (program, period) combo.
    m.RPS_PROGRAM_PERIODS = Set(
        dimen=2,
        within=Any * m.PERIODS,
        initialize=lambda m: unique_list((pr, pe) for pr, pe, z in m.RPS_RULES),
    )

    # Set of all valid program/period/generator combinations (i.e., gens
    # participating in each program in each period). Any eligible gens that are
    # not in a home zone for the RPS program in that period will need to be
    # imported on a bundled or unbundled basis.
    m.RPS_PROGRAM_PERIOD_GENS = Set(
        dimen=3, within=m.RPS_PROGRAM_PERIODS * m.GENERATION_PROJECTS
    )

    m.RPS_PROGRAM_PERIOD_GENS_Have_Rules = m.BuildCheck(
        m.RPS_PROGRAM_PERIOD_GENS,
        rule=lambda m, pr, pe, g: (pr, pe) in m.RPS_PROGRAM_PERIODS,
    )

    # Flags indicating whether each gen is able to send bundled or unbundled
    # RECs to each program it participates in.
    # Note: send_unbundled_recs is mainly used to indicate which regions allow
    # REC trading with each other; send_bundled_recs is mainly used to manage
    # model size, since policy-wise bundled RECs are always allowed, at least in
    # the U.S. (under the dormant commerce clause), but tracking flows between
    # zones is computationally intensive, so it generally works best to only
    # send bundled RECs to neighboring states or major trading partners (if
    # multiple neighboring states have RPSs, state A could send bundled RECs to
    # state B, which then sends some of its own bundled RECs to state C,
    # modeling somewhat wider inter-zone trading). Note that if a gen is able to
    # send bundled RECs to a particular zone for one program, it can also send
    # it to the same zone for other programs, even if not permitted here (BREC
    # transfer is treated as gen-to-zone, not gen-to-program).
    m.send_bundled_recs = Param(m.RPS_PROGRAM_PERIOD_GENS, within=Binary, default=False)
    m.send_unbundled_recs = Param(
        m.RPS_PROGRAM_PERIOD_GENS, within=Binary, default=True
    )

    @m.BuildAction()
    def build_rps_topology_action(m):
        m.rps_topology = build_rps_policy_topology(m)

    # Initialize topology sets from m.rps_topology (there are a lot of them)
    m.ZONES_IN_RPS_PROGRAM_PERIOD = Set(
        m.RPS_PROGRAM_PERIODS,
        within=m.LOAD_ZONES,
        initialize=lambda m, pr, pe: m.rps_topology.zones_in_rps_program_period[pr, pe],
    )

    # Flag identifying whether each gen can create RECs directly (locally) for
    # each RPS program it participates in during each period.
    m.create_local_recs = Param(
        m.RPS_PROGRAM_PERIOD_GENS,
        within=Binary,
        initialize=lambda m, pr, pe, g: int(
            m.rps_topology.create_local_recs[pr, pe, g]
        ),
    )

    # all RECs must be eligible for local or export use, but not both
    m.recs_for_local_xor_export_use = m.BuildCheck(
        m.RPS_PROGRAM_PERIOD_GENS,
        rule=lambda m, pr, pe, g: bool(m.create_local_recs[pr, pe, g])
        ^ bool(m.send_bundled_recs[pr, pe, g] or m.send_unbundled_recs[pr, pe, g]),
    )

    #############
    # Bundled REC (BREC) flows. These are cases where power is scheduled to flow
    # into an RPS's region from an RPS-eligible generator in a zone outside that
    # region. We pre-identify the lowest-loss routes and assume power will
    # always be scheduled to flow along those. Then we require that a matching
    # amount of power is dispatched along these lines. Note that this means
    # bundled RECs can't be applied to one program in one zone and a different
    # program in another zone.
    # Note: in the code below, "BREC-eligible gens" means gens that are
    # eligible to deliver bundled RECs to other zones for RPS programs there.

    # **Eligibility groups** (EGs) are unique combinations of local program,
    # remote bundled and remote unbundled program that power can go to from each
    # zone and period (found by scanning and aggregating eligibility for all
    # gens in the zone and period). **Requirement groups** (RGs) are unique
    # combinations of programs that must be met in some zone during each period
    # (found by scanning across and aggregating zones).

    # EGs are sources for RECs and RGs are potential destinations for RECs.
    # BREC routes connect each EG to all RGs that include any BREC-eligible
    # program in that EG. UREC routes connect each EG to any RGs that include
    # any UREC-eligible program in that EG.

    # BREC flows are scheduled on each route every timepoint. UREC flows are
    # scheduled on each UREC route per period. Anything left over each period is
    # considered a local REC.

    # IDs for eligibility groups; each of these represents a unique collection
    # of programs (and REC methods) that one or more generation projects are
    # eligible for, e.g., 'ESR_AZ_rps local / ESR_CA_ces BREC / ESR_CA_rps BREC'
    m.ELIGIBILITY_GROUPS = Set(
        within=Any, initialize=lambda m: m.rps_topology.eligibility_groups
    )

    # valid combos of load zone, eligibility group and period
    # (may or may not have REC trading allowed), e.g.,
    # ('p27', 'ESR_AZ_rps local / ESR_CA_ces BREC / ESR_CA_rps BREC', 2030), ...
    m.ZONE_ELIGIBILITY_GROUP_PERIODS = Set(
        dimen=3,
        within=m.LOAD_ZONES * m.ELIGIBILITY_GROUPS * m.PERIODS,
        initialize=lambda m: m.rps_topology.zone_eligibility_group_periods,
    )

    # list of gens in zone z that are part of eligibility group eg in period pe
    # (each gen is in exactly one zone-eligibility group per period, so total
    # RECs produced for that group of programs = total production from all gens
    # in the group). e.g.,
    # ('p27', 'ESR_AZ_rps local / ESR_CA_ces BREC / ESR_CA_rps BREC', 2030):
    #   {'p27_conventional_hydroelectric_1', 'p27_landbasedwind_class3_conservative_1', ...}
    m.GENS_IN_ZONE_ELIGIBILITY_GROUP_PERIOD = Set(
        m.ZONE_ELIGIBILITY_GROUP_PERIODS,
        within=m.GENERATION_PROJECTS,
        initialize=lambda m, z, eg, pe: (
            m.rps_topology.gens_in_zone_eligibility_group_period[z, eg, pe]
        ),
    )

    # names of all requirements groups (sets of RPS/CES programs with the same
    # geographic coverage in a period); these are the destinations for BREC and
    # UREC trade routes
    # e.g., 'ESR_CA_ces / ESR_CA_rps' <- CES and RPS in CA will be enforced
    # over the same generators, and imported BRECs and/or URECs may be applied
    # toward this.
    m.REQUIREMENTS_GROUPS = Set(
        dimen=1, within=Any, initialize=lambda m: m.rps_topology.requirements_groups
    )
    # valid combinations of requirements group and period
    m.REQUIREMENTS_GROUP_PERIODS = Set(
        dimen=2,
        within=m.REQUIREMENTS_GROUPS * m.PERIODS,
        initialize=lambda m: m.rps_topology.requirements_group_periods,
    )
    # Set of zones that are part of each requirements group in each period
    # (have the same RPS requirements in that period)
    m.ZONES_IN_REQUIREMENTS_GROUP_PERIOD = Set(
        m.REQUIREMENTS_GROUP_PERIODS,
        within=m.LOAD_ZONES,
        initialize=lambda m, rg, pe: (
            m.rps_topology.zones_in_requirements_group_period[rg, pe]
        ),
    )

    # Set of BREC- or UREC-eligible routes from eligibility groups to matching
    # requirements groups. Per-timepoint BREC flows and per-period UREC flows
    # will be scheduled for each of these. For BRECs, a least-loss set of
    # transmission lines will also be pre-selected corresponding to this route,
    # and losses and congestion will be calculated along that path. Each route
    # is a tuple of (zone, eligibility group, requirements group, period). A
    # requirements group is considered to match an eligibility group in a period
    # if any program in the requirements group is in the eligibility group in
    # that period. This way, the model can choose to send RECs from zone A to
    # any requirements group (collection of zones with the same RPS programs)
    # where they could be useful.
    m.BREC_ROUTES = Set(
        dimen=4,
        # source zone, source eligibility group, destination requirements group, period
        within=m.LOAD_ZONES * m.ELIGIBILITY_GROUPS * m.REQUIREMENTS_GROUPS * m.PERIODS,
        initialize=lambda m: m.rps_topology.brec_routes,
    )
    m.UREC_ROUTES = Set(
        dimen=4,
        within=m.LOAD_ZONES * m.ELIGIBILITY_GROUPS * m.REQUIREMENTS_GROUPS * m.PERIODS,
        initialize=lambda m: m.rps_topology.urec_routes,
    )

    # Set of all BREC and UREC routes that start from each
    # zone/eligibility-group/period combo.
    m.BREC_ROUTES_FOR_ZONE_ELIGIBILITY_GROUP_PERIOD = Set(
        m.ZONE_ELIGIBILITY_GROUP_PERIODS,
        dimen=4,
        within=m.BREC_ROUTES,
        initialize=lambda m, z, eg, pe: (
            m.rps_topology.brec_routes_for_zone_eligibility_group_period[z, eg, pe]
        ),
    )
    m.UREC_ROUTES_FOR_ZONE_ELIGIBILITY_GROUP_PERIOD = Set(
        m.ZONE_ELIGIBILITY_GROUP_PERIODS,
        dimen=4,
        within=m.UREC_ROUTES,
        initialize=lambda m, z, eg, pe: (
            m.rps_topology.urec_routes_for_zone_eligibility_group_period[z, eg, pe]
        ),
    )

    # Set of BREC or UREC routes that serve each RPS program/period.
    m.BREC_ROUTES_FOR_RPS_PROGRAM_PERIOD = Set(
        m.RPS_PROGRAM_PERIODS,
        dimen=4,
        within=m.BREC_ROUTES,
        initialize=lambda m, pr, pe: (
            m.rps_topology.brec_routes_for_rps_program_period[pr, pe]
        ),
    )
    m.UREC_ROUTES_FOR_RPS_PROGRAM_PERIOD = Set(
        m.RPS_PROGRAM_PERIODS,
        dimen=4,
        within=m.UREC_ROUTES,
        initialize=lambda m, pr, pe: (
            m.rps_topology.urec_routes_for_rps_program_period[pr, pe]
        ),
    )

    # Set of zone/eligibility-group/period combos that are local to each RPS
    # program/period, i.e., have generators in a zone governed by the program in
    # that period that can produce RECs for the program.
    m.LOCAL_ZONE_ELIGIBILITY_GROUP_PERIODS_FOR_RPS_PROGRAM_PERIOD = Set(
        m.RPS_PROGRAM_PERIODS,
        dimen=3,
        within=m.ZONE_ELIGIBILITY_GROUP_PERIODS,
        initialize=lambda m, pr, pe: (
            m.rps_topology.local_zone_eligibility_group_periods_for_rps_program_period[
                pr, pe
            ]
        ),
    )

    ###########
    # BREC production during each timepoint

    # amount of BRECs to send from each EG to each eligible RG during each
    # timepoint
    m.BREC_ROUTE_TIMEPOINTS = Set(
        dimen=5,
        within=m.LOAD_ZONES
        * m.ELIGIBILITY_GROUPS
        * m.REQUIREMENTS_GROUPS
        * m.PERIODS
        * m.TIMEPOINTS,
        initialize=lambda m: [
            (z, eg, rg, pe, tp)
            for z, eg, rg, pe in m.BREC_ROUTES
            for tp in m.TPS_IN_PERIOD[pe]
        ],
    )
    m.ExportBRECsTP = Var(m.BREC_ROUTE_TIMEPOINTS, within=NonNegativeReals)

    # exports must be less than production for the corresponding eligibility
    # groups in the source zone/period
    m.ZONE_ELIGIBILITY_GROUP_PERIOD_TIMEPOINTS = Set(
        dimen=4,
        within=m.LOAD_ZONES * m.ELIGIBILITY_GROUPS * m.PERIODS * m.TIMEPOINTS,
        initialize=lambda m: [
            (z, eg, pe, tp)
            for z, eg, pe in m.ZONE_ELIGIBILITY_GROUP_PERIODS
            for tp in m.TPS_IN_PERIOD[pe]
        ],
    )

    @m.Constraint(m.ZONE_ELIGIBILITY_GROUP_PERIOD_TIMEPOINTS)
    def BREC_Export_Below_Production(m, z, eg, pe, tp):
        RTS = m.BREC_ROUTES_FOR_ZONE_ELIGIBILITY_GROUP_PERIOD[z, eg, pe]
        if RTS:
            brecs = sum(
                m.ExportBRECsTP[_z, _eg, _rg, _pe, tp] for _z, _eg, _rg, _pe in RTS
            )
            production = sum(
                m.DispatchGen[g, tp]
                for g in m.GENS_IN_ZONE_ELIGIBILITY_GROUP_PERIOD[z, eg, pe]
                if (g, tp) in m.GEN_TPS
            )
            return brecs <= production
        else:
            return Constraint.Skip

    ###########
    # Find lowest-loss transmission path for each BREC route and require time-
    # matched flows (and losses) along these routes when BRECs are transferred

    @m.BuildAction()
    def build_brec_transmission_topology_action(m):
        m.brec_tx_topology = build_brec_transmission_topology(m)

    m.ZONES_ON_BREC_ROUTE = Set(
        m.BREC_ROUTES,
        within=m.LOAD_ZONES,
        ordered=True,
        initialize=lambda m, z, eg, rg, pe: m.brec_tx_topology.zones_on_brec_route[
            z, eg, rg, pe
        ],
    )

    # set of all zones along any BREC route (just used to index
    # brec_route_efficiency_to_zone)
    m.BREC_ROUTE_ZONES = Set(
        dimen=5,
        ordered=True,
        within=m.BREC_ROUTES * m.LOAD_ZONES,
        initialize=lambda m: m.brec_tx_topology.brec_route_zones,
    )

    # cumulative efficiency up to every zone on every route, including source
    # and destination
    m.brec_route_efficiency_to_zone = Param(
        m.BREC_ROUTE_ZONES,
        initialize=lambda m, z_source, eg, rg, pe, z: (
            m.brec_tx_topology.brec_route_efficiency_to_zone[z_source, eg, rg, pe, z]
        ),
    )

    # BREC routes that use each transmission line in each period
    m.BREC_ROUTES_USING_DIRECTIONAL_TX_IN_PERIOD = Set(
        m.DIRECTIONAL_TX,
        m.PERIODS,
        dimen=4,
        within=m.BREC_ROUTES,
        initialize=lambda m, z_from, z_to, pe: (
            m.brec_tx_topology.brec_routes_using_directional_tx_in_period[
                z_from, z_to, pe
            ]
        ),
    )

    m.BREC_TX_PERIOD_TIMEPOINTS = Set(
        dimen=4,
        within=m.DIRECTIONAL_TX * m.PERIODS * m.TIMEPOINTS,
        initialize=lambda m: m.brec_tx_topology.brec_tx_period_timepoints,
    )

    # for every trans line, BREC trade along routes using that line must not
    # exceed actual power transfers along that line
    @m.Constraint(m.BREC_TX_PERIOD_TIMEPOINTS)
    def Require_BRECs_Below_TX_Transfers(m, z_from, z_to, pe, tp):
        return sum(
            # bundled RECs reaching zone z_from along all z_start ->
            # z_dest routes that use this corridor, net of losses
            # prior to z_from
            m.ExportBRECsTP[rte + (tp,)]
            * m.brec_route_efficiency_to_zone[rte + (z_from,)]
            for rte in m.BREC_ROUTES_USING_DIRECTIONAL_TX_IN_PERIOD[z_from, z_to, pe]
        ) <= (
            # power flow along this corridor
            m.DispatchTx[z_from, z_to, tp]
        )

    ##########
    # Total BREC trade per period for use in program-balancing below

    # amount of BRECs (MWh) exported from each EG to each RG during each period
    m.ExportBRECs = Expression(
        m.BREC_ROUTES,
        rule=lambda m, z, eg, rg, pe: sum(
            m.ExportBRECsTP[z, eg, rg, pe, tp] * m.tp_weight[tp]
            for tp in m.TPS_IN_PERIOD[pe]
        ),
    )

    # amount of BRECs (MWh) reaching each program during each period, net of losses
    m.ImportBRECs = Expression(
        m.RPS_PROGRAM_PERIODS,
        rule=lambda m, pr, pe: sum(
            m.ExportBRECs[rte]
            * m.brec_route_efficiency_to_zone[
                rte + (m.ZONES_ON_BREC_ROUTE[rte].last(),)
            ]
            for rte in m.BREC_ROUTES_FOR_RPS_PROGRAM_PERIOD[pr, pe]
        ),
    )

    ################
    # Unbundled REC (UREC) trading

    # This is similar to bundled REC (BREC) trading (RECs from specific gens are
    # designated as being exported to a particular zone where they are eligible
    # to participate in one or more RPS programs). However, we only tabulate
    # once per period instead of per timepoint, and we don't worry about the
    # transmission lines the power flows on. Note that this disallows the same
    # REC from being used for different programs in different regions
    # (corresponding to requirements groups); we assume each REC moves to one
    # specific requirements group.

    # balance unbundled trades for every project every period: assign MWh from
    # that project to every allowed export jurisdiction, s.t., constraint that
    # allocations to all jurisdictions must be less than or equal to total
    # production minus bundled assignments

    # URECs go along UREC_ROUTES, which are valid combos of source zone,
    # eligibility group, destination requirements group and period for zones
    # outside the home zone of the requirements (i.e., where m.gen_load_zone[g]
    # not in m.ZONES_IN_RPS_PROGRAM_PERIOD[pr, pe])

    # amount of URECs to export from each source eligibility group to each
    # destination requirements group in each period
    m.ExportURECs = Var(m.UREC_ROUTES, within=NonNegativeReals)

    # decision variable for the total number of URECs to import into each
    # program during each period
    m.ImportURECs = Var(m.RPS_PROGRAM_PERIODS, within=NonNegativeReals)

    # upper limit on URECs; ImportURECs may be less than the number exported to
    # that eligibility group (in order to respect limits on use of URECs for
    # some programs), but must not exceed it. Note that the upper limit does not
    # consider transmission losses. Also, the same MWh can be imported
    # repeatedly into all matching RPS_PROGRAM_PERIODS, e.g., a state's
    # overlapping RPS and CES programs, so the total of ImportURECs may exceed
    # the total of ExportURECs.
    m.ImportURECs_below_ExportURECs = Constraint(
        m.RPS_PROGRAM_PERIODS,
        rule=lambda m, pr, pe: m.ImportURECs[pr, pe]
        <= sum(
            m.ExportURECs[z, eg, rg, _pe]
            for z, eg, rg, _pe in m.UREC_ROUTES_FOR_RPS_PROGRAM_PERIOD[pr, pe]
        ),
    )

    # local REC production for each eligibility group in each zone/period is the
    # difference between total production and UREC + BREC exports.
    def rule(m, z, eg, pe):
        brec_export = sum(
            m.ExportBRECs[z, eg, rg, _pe]
            for z, eg, rg, _pe in m.BREC_ROUTES_FOR_ZONE_ELIGIBILITY_GROUP_PERIOD[
                z, eg, pe
            ]
        )
        urec_export = sum(
            m.ExportURECs[z, eg, rg, _pe]
            for z, eg, rg, _pe in m.UREC_ROUTES_FOR_ZONE_ELIGIBILITY_GROUP_PERIOD[
                z, eg, pe
            ]
        )
        total_dispatch = sum(
            m.DispatchGen[g, tp] * m.tp_weight[tp]
            for g in m.GENS_IN_ZONE_ELIGIBILITY_GROUP_PERIOD[z, eg, pe]
            if (g, pe) in m.GEN_PERIODS
            for tp in m.TPS_IN_PERIOD[pe]
        )
        return total_dispatch - brec_export - urec_export

    m.CreateLocalRECs = Expression(m.ZONE_ELIGIBILITY_GROUP_PERIODS, rule=rule)

    # make sure local REC production is non-negative for each zone + eligibility
    # group, each period, i.e., UREC + BREC + local production does not exceed
    # total power production
    m.Require_Total_RECs_Below_Dispatch = Constraint(
        m.ZONE_ELIGIBILITY_GROUP_PERIODS,
        rule=lambda m, z, eg, pe: m.CreateLocalRECs[z, eg, pe] >= 0,
    )

    # Total local RECs delivered to each program during each period
    m.ConsumeLocalRECs = Expression(
        m.RPS_PROGRAM_PERIODS,
        rule=lambda m, pr, pe: sum(
            m.CreateLocalRECs[z, eg, pe]
            for z, eg, _pe in m.LOCAL_ZONE_ELIGIBILITY_GROUP_PERIODS_FOR_RPS_PROGRAM_PERIOD[
                pr, pe
            ]
        ),
    )

    ##############
    # Overall RPS rules and balancing

    # RPS target (MWh) in each zone for each program during each period. Any
    # production from local dist gen or vpp not eligible for the RPS will
    # instead be used to reduce sales of power below the gross level reported in
    # zone_total_demand_in_period_mwh. That in turn reduces the amount that
    # needs to be met under the RPS. (These will later get pooled across zones
    # to create the total target for the program. There's no real need to define
    # targets per zone, but it is more consistent with PowerGenome and avoids
    # the need to add a per-program-period input file.)
    def rule(m, pr, pe, z):
        # production by ineligible dist gen or VPPs
        non_rps_dg_vpp_output = sum(
            m.DispatchGen[g, tp] * m.tp_weight[tp]
            for g in m.GENS_IN_ZONE[z]
            if (
                (m.gen_is_distributed[g] or m.gen_is_vpp[g])
                and (pr, pe, g) not in m.RPS_PROGRAM_PERIOD_GENS
            )
            for tp in m.TPS_FOR_GEN_IN_PERIOD[g, pe]
        )

        # rps share of net load in this zone for this program/period
        return m.rps_share[pr, pe, z] * (
            m.zone_total_demand_in_period_mwh[z, pe] - non_rps_dg_vpp_output
        )

    m.RPSZonalTargetMWh = Expression(m.RPS_RULES, rule=rule)

    # Total RPS target (MWh) for each program during each period.
    m.RPSProgramTargetMWh = Expression(
        m.RPS_PROGRAM_PERIODS,
        rule=lambda m, pr, pe: sum(
            m.RPSZonalTargetMWh[pr, pe, z]
            for z in m.ZONES_IN_RPS_PROGRAM_PERIOD[pr, pe]
        ),
    )

    # Enforce limit on URECs in each RPS program (has to build up from zonal
    # level because unbundled_rec_limit_fraction is specified per zone even
    # though the rule applies per program)
    m.URECs_Below_RPS_Program_Limit = Constraint(
        m.RPS_PROGRAM_PERIODS,
        rule=lambda m, pr, pe: m.ImportURECs[pr, pe]
        <= sum(
            m.unbundled_rec_limit_fraction[pr, pe, z] * m.RPSZonalTargetMWh[pr, pe, z]
            for z in m.ZONES_IN_RPS_PROGRAM_PERIOD[pr, pe]
        ),
    )

    # enforce overall RPS balance, including URECs, BRECs and local REC production
    m.Enforce_RPS_Share = Constraint(
        m.RPS_PROGRAM_PERIODS,
        rule=lambda m, pr, pe: m.ImportBRECs[pr, pe]
        + m.ImportURECs[pr, pe]
        + m.ConsumeLocalRECs[pr, pe]
        >= m.RPSProgramTargetMWh[pr, pe],
    )

    # drop topology caches to save memory
    @m.BuildAction()
    def drop_topology_caches_action(m):
        del m.rps_topology
        del m.brec_tx_topology


def load_inputs(m, switch_data, inputs_dir):
    """
    Optional input files (no RPS/CES will be applied if not supplied); optional
    columns are marked with "*":

        rps_generators.csv
            RPS_PROGRAM, PERIOD, GENERATION_PROJECT, send_bundled_recs*, send_unbundled_recs*

        rps_requirements.csv:
           RPS_PROGRAM, PERIOD, LOAD_ZONE, rps_share, unbundled_rec_limit_fraction*

        gen_info.csv
            gen_is_vpp*
    """

    switch_data.load_aug(
        filename=os.path.join(inputs_dir, "rps_requirements.csv"),
        optional=True,  # also enables empty files
        index=m.RPS_RULES,
        param=(m.rps_share, m.unbundled_rec_limit_fraction),
    )

    switch_data.load_aug(
        filename=os.path.join(inputs_dir, "rps_generators.csv"),
        optional=True,  # also enables empty files
        index=m.RPS_PROGRAM_PERIOD_GENS,
        param=(m.send_bundled_recs, m.send_unbundled_recs),
    )

    # load gen_is_vpp if it was created by this module
    if (
        hasattr(m, "gen_is_vpp")
        and hasattr(m.gen_is_vpp, "added_by")
        and m.gen_is_vpp.added_by == __name__
    ):
        switch_data.load_aug(
            filename=os.path.join(inputs_dir, "gen_info.csv"),
            param=(m.gen_is_vpp,),
        )
