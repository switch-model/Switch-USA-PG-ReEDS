"""
This module minimizes excess renewable production (dissipated in transmission
and battery losses) and smoothes out demand response, EV charging and generator
dispatch as much as possible. It also prevents excess allocation of surplus
reserves. This creates more accurate curtailment estimates and dispatch
schedules with less random variation from hour to hour.

Simple use: add this to modules.txt, below most modules but before reporting.
This module will re-solve the model once to smooth dispatch before moving on to
the reporting code.

Advanced use: add this to modules.txt (anywhere) and also to iterate.txt. This
will use Switch's iterated model mechanism to re-solve once (as a second
iteration) before moving to post-solve, so it should automatically apply
smoothing before all reporting.
"""

from pyomo.environ import *
from pyomo.core.base.numvalue import native_numeric_types
import switch_model.solve
from switch_model.utilities import wrap


def define_components(m):
    # $/MW cost to apply to any jumps in values of the smoothed variables (e.g.,
    # generator output); should be small relative to the cost of generation (e.g.,
    # 1% of fuel cost per MWh), but not so small it causes numerical difficulties
    m.smoothing_cost = Param(default=0.1)

    # Add an alternative objective function that smoothes out time-shiftable energy
    # sources and sinks. Each component listed below should have timepoint as its final
    # index component. They should also be in order from most-smoothed to least-smoothed.
    components_to_smooth = [
        c  # name of component, not component itself
        for lst in [
            "Distributed_Power_Withdrawals",
            "Zone_Power_Withdrawals",
            "Distributed_Power_Injections",
            "Zone_Power_Injections",
        ]
        if hasattr(m, lst)
        for c in getattr(m, lst)
    ]
    if "StorageNetCharge" in components_to_smooth:
        components_to_smooth.remove("StorageNetCharge")
        components_to_smooth.append("ChargeStorage")
    if "ZoneTotalStorageCharging" in components_to_smooth:
        components_to_smooth.remove("ZoneTotalStorageCharging")
        components_to_smooth.append("ChargeStorage")
    if "ZoneTotalCentralDispatch" in components_to_smooth:
        # drop the zone-level totals and smooth individual generators instead
        components_to_smooth.remove("ZoneTotalCentralDispatch")
        components_to_smooth.remove("ZoneTotalDistributedDispatch")
        components_to_smooth.append("DispatchGen")
    if "WithdrawFromCentralGrid" in components_to_smooth:
        # no need to smooth the transfers from central grid to distribution node
        components_to_smooth.remove("WithdrawFromCentralGrid")
        components_to_smooth.remove("InjectIntoDistributedGrid")

    def add_smoothing_entry(m, d, component, key, weight=1.0):
        """
        Add an entry to the dictionary d of elements to smooth. The entry's
        key is based on component name and specified key, and its value is
        an expression whose absolute value should be minimized to smooth the
        model. The last element of the provided key must be a timepoint, and
        the expression is equal to the value of the component at this
        timepoint minus its value at the previous timepoint.
        """
        tp = key[-1]
        prev_tp = m.TPS_IN_TS[m.tp_ts[tp]].prevw(tp)
        entry_key = (str((component.name,) + key[:-1]), tp)
        entry_val = component[key] - component[key[:-1] + (prev_tp,)]
        d[entry_key] = weight * entry_val

    @m.BuildAction()
    def make_component_smoothing_dict(m):
        m.component_smoothing_dict = dict()
        """Find all components to be smoothed"""
        smoothing_comps = []
        for i, c in enumerate(components_to_smooth):
            weight = 0.9 - 0.4 * (i / len(components_to_smooth))
            try:
                comp = getattr(m, c)
            except AttributeError:
                continue
            if isinstance(comp, Param):
                continue  # Params are fixed and not smoothable
            smoothing_comps.append(c)
            for key in comp:
                add_smoothing_entry(m, m.component_smoothing_dict, comp, key, weight)

        # Tell the user what we'll smooth
        if len(smoothing_comps) == 0:
            m.logger.warning("No components found to smooth.")
        elif len(smoothing_comps) == 1:
            m.logger.info(f"Will smooth {smoothing_comps[0]}.")
        else:
            m.logger.info(
                wrap(
                    f"Will smooth {', '.join(smoothing_comps[:-1])} and {smoothing_comps[-1]}."
                )
            )

    # Force IncreaseSmoothedValue to equal any step-up in a smoothed value
    m.ISV_INDEX = Set(
        dimen=2, initialize=lambda m: list(m.component_smoothing_dict.keys())
    )
    m.IncreaseSmoothedValue = Var(m.ISV_INDEX, within=NonNegativeReals)
    m.Calculate_IncreaseSmoothedValue = Constraint(
        m.ISV_INDEX,
        rule=lambda m, k, tp: m.IncreaseSmoothedValue[k, tp]
        >= m.component_smoothing_dict[k, tp],
    )

    @m.Expression(m.TIMEPOINTS)
    def NonSmoothnessCost(m):
        elements = {tp: [] for tp in m.TIMEPOINTS}
        # minimize production (i.e., maximize curtailment / minimize losses)
        for component in m.Zone_Power_Injections:
            component = getattr(m, component)
            for z in m.LOAD_ZONES:
                for tp in m.TIMEPOINTS:
                    elements[tp].append(component[z, tp])

        # maximize up reserves, which will (a) minimize arbitrary burning off of
        # renewables (e.g., via storage) and (b) give better representation of
        # the amount of reserves actually available
        if hasattr(m, "Spinning_Reserve_Up_Provisions") and hasattr(
            m, "GEN_SPINNING_RESERVE_TYPES"
        ):  # advanced module
            print("Will maximize provision of up reserves.")
            reserve_weight = {
                m.options.contingency_reserve_type: 0.9,
                m.options.regulating_reserve_type: 1.1,
            }
            for comp_name in m.Spinning_Reserve_Up_Provisions:
                component = getattr(m, comp_name)
                for rt, ba, tp in component:
                    elements[tp].append(
                        -0.1 * reserve_weight.get(rt, 1.0) * component[rt, ba, tp]
                    )
        # minimize contingency up reserve requirements to avoid spuriously high
        # contingency requirements (they can be any feasible value above the
        # largest contingency)
        if hasattr(m, "Spinning_Reserve_Up_Requirements") and hasattr(
            m, "GEN_SPINNING_RESERVE_TYPES"
        ):  # advanced module
            print("Will minimize requirement for contingency up reserves.")
            for comp_name in m.Spinning_Reserve_Up_Requirements:
                component = getattr(m, comp_name)
                for rt, ba, tp in component:
                    if rt == m.options.contingency_reserve_type:
                        elements[tp].append(component[rt, ba, tp])

        # minimize absolute value of changes in the smoothed variables
        for (k, tp), v in m.IncreaseSmoothedValue.items():
            elements[tp].append(v)

        expr = {tp: m.smoothing_cost * sum(vals) for tp, vals in elements.items()}
        return expr

    m.Cost_Components_Per_TP.append("NonSmoothnessCost")


# TODO: load smoothing_cost from smooth_dispatch.csv
