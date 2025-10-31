"""
Functions to convert data from PowerGenome for use with Switch
"""

from statistics import mean, mode
import math
import logging
import textwrap
import re

from typing import List, Tuple
import numpy as np
import pandas as pd
import scipy
import coloredlogs

from powergenome.time_reduction import kmeans_time_clustering

km_per_mile = 1.60934


# convenience functions to get first/final keys/values from dicts
# (e.g., first year in a dictionary organized by years)
# note: these use the order of creation, not lexicographic order
def first_key(d: dict):
    return next(iter(d.keys()))


def first_value(d: dict):
    return next(iter(d.values()))


def final_key(d: dict):
    return next(reversed(d.keys()))


def final_value(d: dict):
    return next(reversed(d.values()))

class LogFormatter(coloredlogs.ColoredFormatter):
    """
    Shows a colored log message with a hanging indent.
    Assumes the message is the last element in the log line.
    """
    # precompiled regex for ANSI escape sequences
    ANSI_ESCAPE_RE = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')

    def __init__(self, width=90, **kwargs):
        args = dict(
            # fmt="%(levelname)-7s %(shortname)s %(message)s",
            # fmt="%(levelname)s: %(message)s (%(name)s)",
            # fmt="%(name)s %(shortlevelname)s%(message)s",
            fmt="%(name)s %(message)s",
            level_styles={'warning': {'color': 'red'}, 'info': {'color': None}},
            field_styles={'name': {'color': 'blue'}},
        )
        args.update(kwargs)
        super().__init__(**args)
        self.width = width

    def fill_by_paragraph(self, text, initial_indent='', subsequent_indent='', **kwargs):
        paras = []
        for line in text.splitlines():
            if paras: # after first line
                initial_indent = subsequent_indent
            if line.strip():
                paras.append(
                    textwrap.fill(
                        line, 
                        initial_indent=initial_indent, 
                        subsequent_indent=subsequent_indent, 
                        **kwargs
                    )
                )
            else:
                paras.append("")
        return "\n".join(paras)

    def format(self, record):
        # record.shortlevelname = "" if record.levelname == "INFO" else record.levelname+" "
        # return super().format(record)

        if len(record.name) > 24 and "." in record.name:
            name_parts = record.name.split('.')
            record.name = f"{name_parts[0]}.-.{name_parts[-1]}".ljust(24)
        else:
            record.name = record.name.ljust(24)

        # color the message as normal, then add the level name to the msg part
        # if needed
        colored_message = super().format(record)
        msg = record.getMessage()

        if not msg:
            return colored_message

        if record.levelname != 'INFO':
            # insert the levelname as part of the message (for attention
            # and coloring)
            new_msg = record.levelname + ": " + msg
            colored_message = colored_message.replace(msg, new_msg)
            msg = new_msg

        # # wrap the message if currently single line
        # if not "\n" in msg:
        # Calculate visible prefix length, ignoring ANSI codes
        plain_message = self.ANSI_ESCAPE_RE.sub('', colored_message)
        prefix_len = plain_message.find(msg)
        if prefix_len < 0:
            return colored_message
        
        indent = " " * prefix_len
        # Wrap the uncolored message, then reinsert into colored text; limit
        # second-line indent to no more than 40 (could use 25 to stay aligned
        # with other blocks, but it looks a little better to indent the whole
        # block)
        wrapped = self.fill_by_paragraph(
            msg, width=self.width, initial_indent=indent, subsequent_indent=indent[:40]
        )
        # Remove the first-line indent
        wrapped = wrapped[prefix_len:]
        colored_message = colored_message.replace(msg, wrapped, 1)
        
        return colored_message

def switch_fuel_cost_table(
    aeo_fuel_region_map, fuel_prices, regions, scenario, year_list
):
    """
    Create the fuel_cost input file based on REAM Scenario 178.
    Inputs:
        * aeo_fuel_region_map: has aeo_fuel_regions and the ipm regions within each aeo_fuel_region
        * fuel_prices: output from PowerGenome gc.fuel_prices
        * regions: from settings('model_regions')
        * scenario: filtering the fuel_prices table. Suggest using 'reference' for now.
        * year_list: the periods - 2020, 2030, 2040, 2050.  To filter the fuel_prices year column
    Output:
        the fuel_cost_table
            * load_zone: IPM region
            * fuel: based on PowerGenome fuel_prices table
            * period: based on year_list
            * fuel_cost: based on fuel_prices.price
    """

    ref_df = fuel_prices.copy()
    ref_df = ref_df.loc[
        ref_df["scenario"].isin(scenario)
    ]  # use reference scenario for now
    ref_df = ref_df[ref_df["year"].isin(year_list)]
    ref_df = ref_df.drop(["full_fuel_name", "scenario"], axis=1)

    # loop through aeo_fuel_regions.
    # for each of the ipm regions in the aeo_fuel, duplicate the fuel_prices table while adding ipm column
    fuel_cost = pd.DataFrame(columns=["year", "price", "fuel", "region", "load_zone"])
    data = list()
    for region in aeo_fuel_region_map.keys():
        df = ref_df.copy()
        # lookup all fuels available in this region or with no region specified
        # (generally user-defined fuels added earlier)
        df = df[df["region"].isin({region, ""})]
        for ipm in aeo_fuel_region_map[region]:
            ipm_region = ipm
            df["load_zone"] = ipm_region
            fuel_cost = fuel_cost.append(df)
    #     fuel_cost = fuel_cost.append(data)
    fuel_cost.rename(columns={"year": "period", "price": "fuel_cost"}, inplace=True)
    fuel_cost = fuel_cost[["load_zone", "fuel", "period", "fuel_cost"]]
    fuel_cost["period"] = fuel_cost["period"].astype(int)
    fuel_cost = fuel_cost[fuel_cost["load_zone"].isin(regions)]
    return fuel_cost


def switch_fuels(fuel_prices, fuel_emission_factors):
    """
    Create fuels table using fuel_prices (from gc.fuel_prices) and basing other
    columns on REAM scenario 178
    Output columns
        * fuel: based on the fuels contained in the PowerGenome fuel_prices table
        * co2_intensity: based on REAM scenario 178
    """
    fuels = pd.DataFrame({"fuel": fuel_prices["fuel"].unique()})
    fuels["co2_intensity"] = fuels["fuel"].map(fuel_emission_factors).fillna(0)
    return fuels


def create_dict_plantgen(df, column):
    """
    Create dictionary from two columns, removing na's beforehand
    {plant_gen_id: year}
    """
    df = df[df[column].notna()]
    ids = df["plant_gen_id"].to_list()
    dates = df[column].to_list()
    dictionary = dict(zip(ids, dates))
    return dictionary


def plant_gen_id(df):
    """
    Create unique id for generator by combining plant_id_eia and generator_id
    """
    plant_id_eia = df["plant_id_eia"]
    df["plant_gen_id"] = plant_id_eia.astype(str) + "_" + df["generator_id"].astype(str)
    return df


def gen_info_table(gens, settings):
    """
    Create the gen_info table
    Inputs:
        * gens: from PowerGenome gc.create_all_generators() with some extra data
        * spur_capex_mw_mile: based on the settings file ('transmission_investment_cost')['spur']['capex_mw_mile']
        * cogen_tech, baseload_tech, energy_tech, sched_outage_tech, forced_outage_tech
            - these are user defined dictionaries.  Will map values based on the technology
    Output columns:
        * GENERATION_PROJECT: basing on index
        * gen_tech: based on technology
        * gen_energy_source: based on energy_tech input
        * gen_load_zone: IPM region
        * gen_max_age: based on retirement_age
        * gen_can_retire_early: based on Can_Retire and/or New_Build from PowerGenome
        * gen_is_variable: only solar and wind are true
        * gen_is_baseload: from PowerGenome
        * gen_full_load_heat_rate: based on Heat_Rate_MMBTU_per_MWh from all_gen
        * gen_variable_om: based on Var_OM_Cost_per_MWh_mean from all_gen
        * gen_connect_cost_per_mw: based on spur_capex_mw_mile * spur_miles plus substation cost
        * gen_dbid: same as generation_project
        * gen_scheduled_outage_rate: from PowerGenome
        * gen_forced_outage_rate: from PowerGenome
        * gen_capacity_limit_mw: omitted for new thermal plants; upper limits on new renewables (MW total across all).
        * gen_min_load_fraction: from PowerGenome
        * gen_ramp_limit_up: from PowerGenome
        * gen_ramp_limit_down: from PowerGenome
        * gen_min_uptime: from PowerGenome
        * gen_min_downtime: from PowerGenome
        * gen_startup_om: from PowerGenome
        * gen_is_cogen: from PowerGenome
        * gen_storage_efficiency: from PowerGenome
        * gen_store_to_release_ratio: batteries use 1
        * gen_can_provide_cap_reserves: all 1s
        * gen_self_discharge_rate, gen_storage_energy_to_power_ratio: blanks
        # others specified in settings['gen_info_passthrough']: passed from PowerGenome with specified name

    """
    cols = []
    optional_cols = []

    gen_info = gens.copy().reset_index(drop=True)

    # Make sure GENERATION_PROJECT is the first column (index)
    gen_info["GENERATION_PROJECT"] = gen_info["Resource"]
    cols.append("GENERATION_PROJECT")

    # Calculate gen_connect_cost_per_mw from spur_capex and interconnect_capex_mw
    if "spur_capex" in gen_info.columns:
        spur_capex = gen_info["spur_capex"].fillna(0)
    else:
        spur_capex = pd.Series(0, index=gen_info.index)

    spur_capex_mw_mile = settings.get("transmission_investment_cost")["spur"][
        "capex_mw_mile"
    ]
    if "spur_miles" in gen_info.columns and spur_capex_mw_mile:
        # replace zeros with calculated values (seems like a rare or obsolete case)
        calc_spur_capex = (
            gen_info["region"].map(spur_capex_mw_mile) * gen_info["spur_miles"]
        ).fillna(0)
        spur_capex = spur_capex.where(spur_capex != 0, calc_spur_capex)

    gen_info["gen_connect_cost_per_mw"] = spur_capex + (
        gen_info["interconnect_capex_mw"].fillna(0)
        if "interconnect_capex_mw" in gen_info.columns
        else 0.0
    )
    cols.append("gen_connect_cost_per_mw")

    # Include CO2 pipeline costs as part of connection -- could also be in build capex
    if "co2_pipeline_capex_mw" in gen_info.columns:
        gen_info["gen_connect_cost_per_mw"] += gen_info["co2_pipeline_capex_mw"]

    # gen_amortization_period is optional (Switch will use gen_max_age by
    # default). But we report it if PG provides it, in case the settings use a
    # longer retirement_age to prevent age-based retirement.
    for cap_col in ["Capital_Recovery_Period", "cap_recovery_years"]:  # 0.7.0 / 0.6.3
        if cap_col in gen_info.columns:
            gen_info["gen_amortization_period"] = gen_info[cap_col]
            break
    if "gen_amortization_period" in gen_info.columns:
        # drop zeros, which PG 0.7.0+ gives for existing gens (with $0 CapEx)
        # Switch will use generator lifetime instead
        gen_info["gen_amortization_period"] = gen_info[
            "gen_amortization_period"
        ].replace(0, None)
        cols.append("gen_amortization_period")

    # Infer gen energy source if not provided in the PowerGenome settings
    if "gen_energy_source" not in gen_info.columns:
        gen_info["gen_energy_source"] = infer_gen_energy_source(gen_info, settings)
    cols.append("gen_energy_source")

    # gen_storage_efficiency
    # See documentation for STOR variable at https://github.com/macroenergy/Dolphyn.jl/blob/main/docs/src/data_documentation.md
    # note: GenX has an option for STOR = 2 and then charge capacity becomes
    # a separate decision variable capped by `Max_Charge_Cap_MW`, but Switch
    # doesn't currently handle that, so we treat all storage as symmetrical
    # TODO: add charging equipment cost for STOR=2 cases to discharging
    # CapEx, assuming discharge capacity = charge capacity
    storage_gens = gen_info["STOR"] > 0
    gen_info.loc[storage_gens, "gen_storage_efficiency"] = (
        gen_info["Eff_Up"] * gen_info["Eff_Down"]
    )
    cols.append("gen_storage_efficiency")

    # get capacity limit if any, but ignore -1 (no limit; doesn't seem to be
    # used) or existing plants with 0 limit. PG doesn't seem to create existing
    # plant records that can also be built in the future, and if it did, we
    # would split them into different rows (see notes around
    # `existing_gens["new_build"] = False` in gen_tables()). So this will only
    # affect the existing versions. PG 0.7.0 and maybe later assign 0 for
    # existing plants, which would cause infeasibility in Switch, so we force
    # those to NaN here.
    gen_info["gen_capacity_limit_mw"] = (
        gen_info["Max_Cap_MW"].replace(-1, None)
        if "Max_Cap_MW" in gen_info.columns
        else None
    )
    gen_info.loc[
        gen_info["existing"] & (gen_info["gen_capacity_limit_mw"] == 0),
        "gen_capacity_limit_mw",
    ] = None
    cols.append("gen_capacity_limit_mw")

    # Get capture efficiency for CCS plants if available, omitting non-CCS plants
    if "CO2_Capture_Rate" in gen_info.columns:
        gen_info["gen_ccs_capture_efficiency"] = gen_info["CO2_Capture_Rate"].replace(
            0, None
        )
        cols.append("gen_ccs_capture_efficiency")

    # identify generators that can retire early
    try:
        # settings for newer GenX (not yet implemented as of Aug. 2024)
        gen_info["gen_can_retire_early"] = gen_info["Can_Retire"].astype("Int64")
    except KeyError:
        # settings for older GenX
        # New_Build == -1 -> existing, cannot retire
        # New_Build == 0 -> existing, can retire
        # New_Build >= 1 -> new build, can retire in current version of GenX
        gen_info["gen_can_retire_early"] = (gen_info["New_Build"] >= 0).astype("Int64")
    cols.append("gen_can_retire_early")

    # variable O&M must be filled in
    gen_info["gen_variable_om"] = gen_info["Var_OM_Cost_per_MWh_mean"].fillna(0)
    cols.append("gen_variable_om")

    # both VRE (renewables) and FLEX (demand response) have limiting profiles
    gen_info["gen_is_variable"] = (
        (gen_info["VRE"] > 0) | (gen_info["FLEX"] > 0)
    ).astype(int)
    cols.append("gen_is_variable")

    # Min_Power can be negative in flexible demand cases with load that
    # sometimes goes negative; we ignore that
    if "Min_Power" in gen_info.columns:
        gen_info["gen_min_load_fraction"] = gen_info["Min_Power"].clip(0, None)
        cols.append("gen_min_load_fraction")

    # Keep some columns as-is, just renaming
    rename_cols = {
        "technology": "gen_tech",
        "region": "gen_load_zone",
        "retirement_age": "gen_max_age",
        "Heat_Rate_MMBTU_per_MWh": "gen_full_load_heat_rate",
        "Ramp_Up_Percentage": "gen_ramp_limit_up",
        "Ramp_Dn_Percentage": "gen_ramp_limit_down",
        "Up_Time": "gen_min_uptime",
        "Down_Time": "gen_min_downtime",
        "Start_Cost_per_MW": "gen_startup_om",
        # gen_self_discharge_rate is not defined in main Switch 2.0.10 or earlier. Used by UCSD?
        "Self_Disch": "gen_self_discharge_rate",
        "tonne_co2_captured_mwh": "gen_ccs_load_mwh_per_tCO2",
        # gen_ccs_energy_load was renamed in Switch 2.0.10; this will pick up the old or new name
        "gen_ccs_energy_load": "gen_ccs_load_fraction",
        # If using Switch unit commitment, MUST_RUN will be treated as must run at 100% when committed.
        # If not using unit commitment, MUST_RUN will be treated as must run at same level for the whole period.
        # This behavior probably differs from GenX
        "MUST_RUN": "gen_is_baseload",
        "FLEX": "gen_is_vpp",
    }
    rename_cols.update(
        # pass through extra columns specified in the settings, swapping order to rename
        {pg: sw for (sw, pg) in settings.get("gen_info_extra_columns", {}).items()}
    )
    gen_info.rename(columns=rename_cols, inplace=True)
    cols.extend([c for c in rename_cols.values() if c in gen_info.columns])

    # Add columns for state-level environmental policies
    cols.extend(
        c
        for c in gen_info.columns
        if any(c.startswith(f"{tag}_") for tag in ["ESR", "MinCapTag", "MaxCapTag"])
    )

    # Additional columns that will be passed through as-is if defined in the
    # settings files. Typically these would be defined in
    # pg_settings/extra_inputs/misc_gen_inputs on a per-resource-type basis.
    # Unless otherwise noted, these are not usually defined.
    optional_cols = [
        "gen_dbid",
        "gen_is_cogen",
        "gen_scheduled_outage_rate",
        "gen_forced_outage_rate",
        "gen_store_to_release_ratio",
        "gen_storage_energy_to_power_ratio",
        "gen_can_provide_cap_reserves",
        "gen_type",  # not used by standard Switch, maybe used by UCSD?
    ]

    cols.extend(c for c in optional_cols if c in gen_info.columns)
    gen_info = gen_info[cols]
    return gen_info


def infer_gen_energy_source(gen_info, settings):
    """
    Create a gen_energy_source series based on technology name and standard
    PG generator type flags. First tries to match the technology in
    tech_fuel_map, then falls back to the other approaches.
    """

    # direct lookup of fuel name (simple, but misses some (new build?) technologies)
    # gen_energy_source = gen_info["technology"].map(settings['tech_fuel_map'])

    info = gen_info[["region", "technology", "Fuel"]]

    # first try splitting the generic fuel off the end of the Fuel column
    aeo_region_map = {
        z: a for (a, zones) in settings["aeo_fuel_region_map"].items() for z in zones
    }
    info["aeo_region"] = info["region"].map(aeo_region_map)

    def split_fuel(row):
        if row.Fuel.startswith(row.aeo_region + "_"):
            return row.Fuel[len(row.aeo_region) + 1 :]
        else:
            return None

    info["gen_energy_source"] = info.apply(split_fuel, axis=1)

    # assign energy source based on standard PowerGenome flags
    flag_based_energy_sources = [
        ("STOR", "storage"),
        ("HYDRO", "water"),
        ("FLEX", "demand_response"),
    ]
    for flag, source in flag_based_energy_sources:
        mask = (gen_info[flag] > 0) & info["gen_energy_source"].isna()
        info.loc[mask, "gen_energy_source"] = source

    # Assign remaining generators (wind, solar and a few other no-fuel
    # technologies) by matching terms in their name. Weaker matches are near the
    # end so they'll be tried later.
    non_fuel_energy_sources = [
        ("photovoltaic", "sun"),
        ("solar", "sun"),
        ("wind", "wind"),
        ("hydro", "water"),  # small hydro may not be marked 'HYDRO'
        ("geothermal", "geothermal"),
        ("biomass", "biomass"),
        ("import", "imports"),
        ("demand", "demand_response"),
        ("distributed", "sun"),  # we assume distributed gen is solar
        ("nuclear", "uranium"),
        ("geo", "geothermal"),
        ("pv", "sun"),
        ("resp", "demand_response"),
        ("dr", "demand_response"),
    ]
    tech_non_fuel_map = {
        t: t.lower()  # use tech name as energy source by default
        for t in gen_info.loc[info["gen_energy_source"].isna(), "technology"].unique()
    }
    # map known technology names to specific energy sources
    for tech in tech_non_fuel_map:
        for pattern, fuel in non_fuel_energy_sources:
            if pattern in tech.lower():
                tech_non_fuel_map[tech] = fuel
                break
    info["gen_energy_source"] = info["gen_energy_source"].fillna(
        gen_info["technology"].map(tech_non_fuel_map)
    )
    return info["gen_energy_source"]


hydro_forced_outage_tech = {
    # "conventional_hydroelectric": 0.05,
    # "hydroelectric_pumped_storage": 0.05,
    # "small_hydroelectric": 0.05,
    "conventional_hydroelectric": 0,
    "hydroelectric_pumped_storage": 0,
    "small_hydroelectric": 0,
}


def match_hydro_forced_outage_tech(x):
    for key in hydro_forced_outage_tech:
        if key in x:
            return hydro_forced_outage_tech[key]


def fuel_market_tables(fuel_prices, aeo_fuel_region_map, scenario):
    """
    Create regional_fuel_markets and zone_to_regional_fuel_market
    SWITCH does not seem to like this overlapping with fuel_cost. So all of this might be incorrect.
    """

    # create initial regional fuel market.  Format: region - fuel
    reg_fuel_mar_1 = fuel_prices.copy()
    reg_fuel_mar_1 = reg_fuel_mar_1.loc[
        reg_fuel_mar_1["scenario"] == scenario
    ]  # use reference for now
    reg_fuel_mar_1 = reg_fuel_mar_1.drop(
        ["year", "price", "full_fuel_name", "scenario"], axis=1
    )
    reg_fuel_mar_1 = reg_fuel_mar_1.rename(columns={"region": "regional_fuel_market"})
    reg_fuel_mar_1 = reg_fuel_mar_1[["regional_fuel_market", "fuel"]]

    fuel_markets = reg_fuel_mar_1["regional_fuel_market"].unique()

    # from region to fuel
    group = reg_fuel_mar_1.groupby("regional_fuel_market")
    fuel_market_dict = {}
    for region in fuel_markets:
        df = group.get_group(region)
        fuel = df["fuel"].to_list()
        fuel = list(set(fuel))
        fuel_market_dict[region] = fuel

    # create zone_regional_fuel_market
    data = list()
    for region in aeo_fuel_region_map.keys():
        for i in range(len(aeo_fuel_region_map[region])):
            ipm = aeo_fuel_region_map[region][i]
            for fuel in fuel_market_dict[region]:
                data.append([ipm, ipm + "-" + fuel])

    zone_regional_fm = pd.DataFrame(data, columns=["load_zone", "regional_fuel_market"])

    # use that to finish regional_fuel_markets
    regional_fuel_markets = zone_regional_fm.copy()
    regional_fuel_markets["fuel_list"] = regional_fuel_markets[
        "regional_fuel_market"
    ].str.split("-")
    regional_fuel_markets["fuel"] = regional_fuel_markets["fuel_list"].apply(
        lambda x: x[-1]
    )
    regional_fuel_markets = regional_fuel_markets[["regional_fuel_market", "fuel"]]

    return regional_fuel_markets, zone_regional_fm


def ts_tp_pg_kmeans(
    representative_point: pd.DataFrame,
    point_weights: List[int],
    days_per_period: int,
    planning_year: int,
    planning_start_year: int,
):
    """Create timeseries and timepoints tables when using kmeans time reduction in PG

    Parameters
    ----------
    representative_point : pd.DataFrame
        The representative periods used. Single column dataframe with col name "slot"
    point_weights : List[int]
        The weight assigned to each period. Equal to the number of periods in the year
        that each period represents.
    days_per_period : int
        How long each period lasts in days
    planning_periods : List[int]
        A list of the planning years
    planning_period_start_years : List[int]
        A list of the start year for each planning period, used to calculate the number
        of years in each period

    Returns
    -------
    pd.DataFrame, pd.DataFrame
        A tuple of the timeseries and timepoints dataframes
    """
    ts_data = {
        "timeseries": [],
        "ts_period": [],
        "ts_duration_of_tp": [],
        "ts_num_tps": [],
        "ts_scale_to_period": [],
    }
    tp_data = {
        "timestamp": [],
        "timeseries": [],
    }
    planning_yrs = planning_year - planning_start_year + 1
    for p, weight in zip(representative_point, point_weights):
        num_hours = days_per_period * 24
        ts = f"{planning_year}_{p}"
        ts_data["timeseries"].append(ts)
        ts_data["ts_period"].append(planning_year)
        ts_data["ts_duration_of_tp"].append(1)
        ts_data["ts_num_tps"].append(num_hours)
        ts_data["ts_scale_to_period"].append(weight * planning_yrs)

        tp_data["timestamp"].extend([f"{ts}_{i}" for i in range(num_hours)])
        tp_data["timeseries"].extend([ts for i in range(num_hours)])

    timeseries = pd.DataFrame(ts_data)
    timepoints = pd.DataFrame(tp_data)
    timepoints["timepoint_id"] = timepoints.index + 1
    timepoints = timepoints[["timepoint_id", "timestamp", "timeseries"]]
    return timeseries, timepoints


def hydro_timepoints_pg_kmeans(timepoints: pd.DataFrame) -> pd.DataFrame:
    """Create the timepoints table when using kmeans time reduction in PG

    This assumes that the hydro timeseries are identical to the model timeseries.

    Parameters
    ----------
    timepoints : pd.DataFrame
        The timepoints table

    Returns
    -------
    pd.DataFrame
        Identical to the incoming timepoints table except "timepoint_id" is renamed to
        "tp_to_hts"
    """

    hydro_timepoints = timepoints.copy()
    hydro_timepoints = hydro_timepoints.rename(columns={"timeseries": "tp_to_hts"})

    return hydro_timepoints[["timepoint_id", "tp_to_hts"]]


def hydro_timeseries_pg_kmeans(
    gen: pd.DataFrame,
    hydro_variability: pd.DataFrame,
    hydro_timepoints: pd.DataFrame,
    outage_rate: float = 0,
) -> pd.DataFrame:
    """Create hydro timeseries table when using kmeans time reduction in PG

    The hydro timeseries table has columns hydro_project, timeseries, outage_rate,
    hydro_min_flow_mw, and hydro_avg_flow_mw. The "timeseries" column links to the
    column "tp_to_hts" in hydro_timepoints.csv. "hydro_min_flow_mw" uses the resource
    minimum capacity (calculated in PG from EIA860). "hydro_avg_flow_mw" is the average
    of flow during each timeseries.

    Parameters
    ----------
    existing_gen : pd.DataFrame
        All existing generators, one row per generator. Columns must include "Resource",
        "Existing_Cap_MW", "Min_Power", and "HYDRO".
    hydro_variability : pd.DataFrame
        Hourly flow/generation capacity factors. Should have column names that correspond
        to the "Resource" column in `existing_gen`. Additional column names will be
        filtered out.
    hydro_timepoints : pd.DataFrame
        All timepoints for hydro, with the column "tp_to_hts"
    outage_rate : float, optional
        The average outage rate for hydro generators, by default 0.05

    Returns
    -------
    pd.DataFrame
        The hydro_timeseries table for Switch
    """

    hydro_df = gen.copy()
    # ? why multiply Min_Power
    # hydro_df["min_cap_mw"] = hydro_df["Existing_Cap_MW"] * hydro_df["Min_Power"]
    hydro_df = hydro_df.loc[hydro_df["HYDRO"] == 1, :]

    hydro_variability = hydro_variability.loc[:, hydro_df["Resource"]]

    # for col in hydro_variability.columns:
    #     hydro_variability[col] *= hydro_df.loc[
    #         hydro_df["Resource"] == col, "Existing_Cap_MW"
    #     ].values[0]
    hydro_variability["timeseries"] = hydro_timepoints["tp_to_hts"].values
    hydro_ts = hydro_variability.melt(id_vars=["timeseries"])
    hydro_ts = hydro_ts.groupby(["timeseries", "Resource"], as_index=False).agg(
        hydro_avg_flow_mw=("value", "mean"), hydro_min_flow_mw=("value", "min")
    )

    # hydro_ts["hydro_min_flow_mw"] = hydro_ts["Resource"].map(
    #     hydro_df.set_index("Resource")["Min_Power"]
    # )
    hydro_ts["hydro_avg_flow_mw"] = hydro_ts["hydro_avg_flow_mw"] * hydro_ts[
        "Resource"
    ].map(hydro_df.set_index("Resource")["Existing_Cap_MW"])
    hydro_ts["hydro_min_flow_mw"] = hydro_ts["hydro_min_flow_mw"] * hydro_ts[
        "Resource"
    ].map(hydro_df.set_index("Resource")["Existing_Cap_MW"])

    hydro_ts["outage_rate"] = outage_rate
    hydro_ts = hydro_ts.rename(columns={"Resource": "hydro_project"})
    cols = [
        "hydro_project",
        "timeseries",
        # "outage_rate",
        "hydro_min_flow_mw",
        "hydro_avg_flow_mw",
    ]
    return hydro_ts[cols]


# # obsolete, ended up identical to variable_capacity_factors_table
# def variable_cf_pg_kmeans(
#     all_gens: pd.DataFrame, all_gen_variability: pd.DataFrame, timepoints: pd.DataFrame
# ) -> pd.DataFrame:
#     """Create the variable capacity factors table when using kmeans time reduction in PG

#     Variable generators are identified as those with hourly average capacity factors
#     less than 1.

#     Parameters
#     ----------
#     all_gens : pd.DataFrame
#         All resources. Must have the columns "Resource" and "VRE".
#     all_gen_variability : pd.DataFrame
#         Wide dataframe with hourly capacity factors of all generators.
#     timepoints : pd.DataFrame
#         Timepoints table with column "timepoint_id"

#     Returns
#     -------
#     pd.DataFrame
#         Tidy dataframe with columns "GENERATION_PROJECT", "timepoint", and
#         "gen_max_capacity_factor"
#     """

#     vre_gens = all_gens.loc[all_gens["VRE"] == 1, "Resource"]
#     vre_variability = all_gen_variability[vre_gens]
#     vre_variability["timepoint_id"] = timepoints["timepoint_id"].values
#     vre_ts = vre_variability.melt(
#         id_vars=["timepoint_id"], value_name="gen_max_capacity_factor"
#     )
#     vre_ts = vre_ts.rename(
#         columns={"Resource": "GENERATION_PROJECT", "timepoint_id": "timepoint"}
#     )

#     return vre_ts.reindex(
#         columns=["GENERATION_PROJECT", "timepoint", "gen_max_capacity_factor"]
#     )


def load_pg_kmeans(load_curves: pd.DataFrame, timepoints: pd.DataFrame) -> pd.DataFrame:
    """Create the loads table when using kmeans time reduction in PG

    Parameters
    ----------
    load_curves : pd.DataFrame
        Wide dataframe with one column of demand values for each zone
    timepoints : pd.DataFrame
        Timepoints table with column "timepoint_id"

    Returns
    -------
    pd.DataFrame
        Tidy dataframe with columns "LOAD_ZONE" and "TIMEPOINT"
    """
    load_curves = load_curves.astype(int)
    load_curves.columns.name = "LOAD_ZONE"  # rename from "region" or None
    load_curves["TIMEPOINT"] = timepoints["timepoint_id"].values
    load_ts = load_curves.melt(id_vars=["TIMEPOINT"], value_name="zone_demand_mw")
    load_ts["zone_demand_mw"] = load_ts["zone_demand_mw"].astype("object")

    # change the order of the columns
    return load_ts[["LOAD_ZONE", "TIMEPOINT", "zone_demand_mw"]]


def graph_timestamp_map_kmeans(timepoints_df):
    """
    Create the graph_timestamp_map table for Switch-WECC (UCSD)
    Input:
        timeseries_df, timepoints_df: the SWITCH timeseries table
    Output columns:
        * timestamp: dates based on the timeseries table
        * time_row: the period decade year based on the timestamp
        * time_column: format: yyyymmdd. Using 2012 because that is the year data is based on.
    """

    timepoints_df_copy = timepoints_df.copy()
    graph_timeseries_map = pd.DataFrame(columns=["timestamp", "time_row", "timeseries"])
    graph_timeseries_map["timestamp"] = timepoints_df_copy["timestamp"]
    graph_timeseries_map["timeseries"] = timepoints_df_copy["timeseries"]
    graph_timeseries_map["time_row"] = [
        x[0] for x in graph_timeseries_map["timestamp"].str.split("_")
    ]

    # using 2012 for financial year
    graph_timeseries_map["time_column"] = graph_timeseries_map["timeseries"].apply(
        lambda x: str(2012) + x[5:]
    )

    return graph_timeseries_map


def make_n_timepoints(num_steps, planning_year):
    """
    Generate a pandas Series with unique timepoint IDs for the specified number
    of steps in the specified planning year. These are integers that encode the
    year, possibly the repetition count, month, day and hour. We assume the steps
    are hourly and cover an integer number of years.
    """
    if num_steps % 8760 != 0:
        raise ValueError(
            "`n_steps` must be an exact multiple of 8760 for make_n_timepoints()"
        )

    dates = [
        d.strftime("%Y%m%d")
        for d in pd.date_range(f"{planning_year}-01-01", f"{planning_year}-12-31")
    ]
    # ignore leap day if present in the model year, since we always use n x 8760 sample hours
    leap_yr = f"{planning_year:04d}0229"
    if leap_yr in dates:
        dates.remove(leap_yr)

    num_years = num_steps // 8760
    if num_years > 1:
        # repeat the year n_years times and insert a repetition number
        digits = int(math.log10(num_years)) + 1
        dates = [
            # 2030020301 = year 2030, repetition 02, month 03, day 01
            f"{d[:4]}{r+1:0{digits}d}{d[4:]}"
            for r in range(num_years)
            for d in dates
        ]

    hours = [f"{i:02d}" for i in range(24)]

    return pd.Series(f"{d}{h}" for d in dates for h in hours)


def timeseries_full(
    planning_year,
    planning_start_year,
    num_sample_hours,
) -> Tuple[
    pd.DataFrame,  # timeseries_df
    pd.DataFrame,  # timepoints_df
]:
    """
    Create timeseries and timepoints tables when using data with 8760 hours per
    year for one or more years. Apply this function when reduce_time_domain is
    False in settings.

    Parameters
    ----------
    planning_year:
        model year to prepare data for (assumed to be the end of the period)
    planning_period_start_year:
        first year of the study period, used to calculate the number
        of years in the period
    num_sample_hours:
        number of sample hours to generate timeseries for
    settings:
        PowerGenome settings dict

    Returns
    -------
    pd.DataFrame, pd.DataFrame, pd.DataFrame
        A tuple of the timeseries and timepoints dataframes
    """
    timepoints = make_n_timepoints(num_sample_hours, planning_year)

    num_hours = len(timepoints)
    sample_to_year_ratio = 8760 / num_hours
    num_planning_yrs = planning_year - planning_start_year + 1

    # define a single timeseries for all the hours
    ts = f"{planning_year}-full"
    timeseries_df = pd.DataFrame(
        {
            "timeseries": [ts],
            "ts_period": [f"{planning_year}"],
            "ts_duration_of_tp": [1],  # each hour as one timepoint
            "ts_num_tps": [num_hours],
            "ts_scale_to_period": [num_planning_yrs * sample_to_year_ratio],
        }
    )

    timepoints_df = pd.DataFrame(
        {
            "timepoint_id": timepoints,
            "timeseries": [ts] * num_hours,
            "timestamp": timepoints,
        }
    )
    return timeseries_df, timepoints_df


def hydro_time_tables(
    gens: pd.DataFrame,
    gen_variability: pd.DataFrame,
    timepoints_df: pd.DataFrame,
    planning_year,
) -> Tuple[pd.DataFrame, pd.DataFrame]:  # (hydro_timepoints, hydro_timeseries)
    """
    Create the hydro_timepoints and hydro_timeseries tables for the UCSD/WECC version of Switch
    (these won't work with the hydro_simple module from the standard Switch as of mid-2025.)
    Inputs:
        * gens: all generators (resources) active in this period
        * gen_variability: 0-1 availability matrix, one row per timepoint, one column per generator
        * timepoints_df: timepoints to be output; timestamp should be in pattern "YYYY[r]MMDDHH",
          where r may or may not be present, but if it is, it identifies a year repetition number
          when a multi-year timeseries is used for one study year (YYYY)
    Output Columns
        * hydro_timeseries
            * hydro_project: name of a hydro generation project (Resource from gens, with HYDRO==1)
            * timeseries: ID for a month when min and avg flow limits will be enforced;
                  Format: "YYYY[r]MM". Based on the timestamp date from the timepoints table.
            * outage_rate: outage rate for hydro during this time; appears to be unused
            * hydro_min_flow_mw: minimum hydro flow rate (MW equiv) to enforce during this block
            * hydro_avg_flow_mw: average hydro flow rate (MW equiv) to enforce during this block
        * hydro_timepoints
            * timepoint_id: from the timepoints table
            * tp_to_hts: timeseries value from hydro_timeseries, identifies which timepoints
                are in each month
    """

    # only work with HYDRO gens (note: as of 2024, PowerGenome only flags large
    # hydro as "HYDRO" == 1; small hydro is treated as a standard intermittent
    # renewable resource and pumped storage hydro is treated as standard
    # storage)
    hydro_gens = gens.loc[gens["HYDRO"] == 1, :]
    hydro_variability = gen_variability.loc[:, hydro_gens["Resource"]]

    # get cap size for each hydro tech
    hydro_Cap_Size = hydro_gens["Existing_Cap_MW"].to_list()  # cap size for each hydro
    # multiply cap size by hourly
    for i in range(len(hydro_Cap_Size)):
        hydro_variability.iloc[:, i] *= hydro_Cap_Size[i]

    # define hydro_timepoints table, with standard timepoint_ids and tp_to_hts
    # to bridge to hydro_timeseries identifiers
    hydro_timepoints = timepoints_df[["timepoint_id"]]
    # assign a different timeseries for each historical month by dropping the
    # day and hour from the end of the timestamp and keeping the year,
    # year-repetition number and month at the start.
    hydro_timepoints["tp_to_hts"] = timepoints_df["timestamp"].str[:-4]

    # group hydro variability data by hydro timeseries, then aggregate
    hydro_variability["timeseries"] = hydro_timepoints["tp_to_hts"]

    # get minimum and average flow per timeseries per project
    hydro_var_long = hydro_variability.melt(
        id_vars="timeseries", var_name="hydro_project", value_name="MW"
    )
    hydro_timeseries = (
        hydro_var_long.groupby(["hydro_project", "timeseries"])["MW"]
        .agg(hydro_min_flow_mw="mean", hydro_avg_flow_mw="min")
        .reset_index()
    )
    hydro_timeseries["outage_rate"] = hydro_timeseries["hydro_project"].map(
        match_hydro_forced_outage_tech
    )

    return hydro_timepoints, hydro_timeseries


def hydro_system_tables(
    gens,
    gen_variability: pd.DataFrame,
    hydro_timepoints: pd.DataFrame,
    flow_per_mw: float = 1.02,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create the tables specific for module hydro_system
    Inputs:
        1) flow_per_mw: 1/[(1000 kg/m3)(9.8 N/kg)(100 m)(1 MWs/1e6 Nm)] = 1.02 m3/s/MW
        2) reservoir_capacity_m3 <- flow_per_mw * Hydro_Energy_to_Power_Ratio * [generator capacity (MW)]
        3) inflow_m3_per_s <- flow_per_mw * [variable inflow (% of nameplate power)] * [generator capacity (MW)]
    Output tables:
        * water_modes.csv:
            # add these rows, representing the reservoir upstream of each dam
            WATER_NODES <- generator name + “_inlet” (or similar)
            wn_is_sink <- 0
            wnode_constant_consumption <- 0
            wnode_constant_inflow <- 0
        * water_nodes.csv:
            # add a second set of rows to this file, representing reservoir downstream of each dam
             WATER_NODES <- generator name + “_outlet” (or similar)
            wn_is_sink <- 1
            wnode_constant_consumption <- 0
            wnode_constant_inflow <- 0
        * water_connections.csv
            WATER_CONNECTIONS <- generator name
            water_node_from <- generator name + “_inlet”
            water_node_to <- generator name + “_outlet”
        *reservoirs.csv
            RESERVOIRS <- generator name + “_inlet”
            res_min_vol <- 0
            res_max_vol <- reservoir_capacity_m3 (see above)
            # arbitrarily assume reservoir must start and end at 50% full
            initial_res_vol <- 0.5 * reservoir_capacity_m3
            final_res_vol <- 0.5 * reservoir_capacity_m3
        *hydro_generation_projects.csv
            HYDRO_GENERATION_PROJECTS <- generator name (should match gen_info.csv)
            # hydro_efficiency is MW output per m3/s of input
            hydro_efficiency <- 1 / flow_per_mw
            hydraulic_location <- generator name (should match water_connections.csv)
        *water_node_tp_flows.csv
            WATER_NODES <- generator name + “_inlet”
            TIMEPOINTS <- timepoint
            wnode_tp_inflow <- inflow_m3_per_s
            wnode_tp_consumption <- 0
    """

    hydro_gens = gens.query("HYDRO == 1")
    hydro_variability = gen_variability.loc[:, hydro_gens["Resource"]].copy()

    # for water_nodes.csv
    water_nodes_in = pd.DataFrame()
    water_nodes_in["WATER_NODES"] = hydro_gens["Resource"] + "_inlet"
    water_nodes_in["wn_is_sink"] = 0
    water_nodes_in["wnode_constant_consumption"] = 0
    water_nodes_in["wnode_constant_inflow"] = 0
    water_nodes_out = pd.DataFrame()
    water_nodes_out["WATER_NODES"] = hydro_gens["Resource"] + "_outlet"
    water_nodes_out["wn_is_sink"] = 1
    water_nodes_out["wnode_constant_consumption"] = 0
    water_nodes_out["wnode_constant_inflow"] = 0
    water_nodes = pd.concat([water_nodes_in, water_nodes_out])
    # for water_connections.csv
    water_connections = pd.DataFrame()
    water_connections["WATER_CONNECTIONS"] = hydro_gens["Resource"]
    water_connections["water_node_from"] = hydro_gens["Resource"] + "_inlet"
    water_connections["water_node_to"] = hydro_gens["Resource"] + "_outlet"
    water_connections["wc_capacity"] = None  # m3/s, unknown, set unlimited

    # for reservoirs.csv
    # note: reservoir volume and level are given in million m^3
    reservoirs = pd.DataFrame()
    reservoirs["RESERVOIRS"] = hydro_gens["Resource"] + "_inlet"
    reservoirs["res_min_vol"] = 0
    # (MW)(hour)((m3/s)/MW)(3600 sec/hour)(1 million m3/1e6 m3) = million m3
    reservoirs["res_max_vol"] = (
        hydro_gens["Existing_Cap_MW"]  # MW
        * hydro_gens["Hydro_Energy_to_Power_Ratio"]  # h
        * flow_per_mw  # m3/s / MW
        * 3600  # m3/sec -> m3/hour
        * 1e-6  # m3 -> million m3
    )

    # for hydro_generation_projects.csv
    hydro_pj = pd.DataFrame()
    hydro_pj["HYDRO_GENERATION_PROJECTS"] = hydro_gens["Resource"]
    hydro_pj["hydro_efficiency"] = 1 / flow_per_mw
    hydro_pj["hydraulic_location"] = hydro_gens["Resource"]
    # for water_node_tp_flows.csv
    hydro_variability["TIMEPOINTS"] = hydro_timepoints["timepoint_id"].values
    water_node_tp = hydro_variability.melt(id_vars=["TIMEPOINTS"])
    water_node_tp["wnode_tp_inflow"] = (
        flow_per_mw
        * water_node_tp["value"]
        * water_node_tp["Resource"].map(
            hydro_gens.set_index("Resource")["Existing_Cap_MW"]
        )
    )
    water_node_tp["WATER_NODES"] = water_node_tp["Resource"] + "_inlet"
    water_node_tp["wnode_tp_consumption"] = 0
    cols = [
        "WATER_NODES",
        "TIMEPOINTS",
        "wnode_tp_inflow",
        "wnode_tp_consumption",
    ]
    water_node_tp_flows = water_node_tp[cols]

    return water_nodes, water_connections, reservoirs, hydro_pj, water_node_tp_flows


def graph_timestamp_map_table(timepoints_df):
    """
    Create the graph_timestamp_map table for Switch-WECC (UCSD)
    Input:
        1) timeseries_df: the SWITCH timeseries table
        2) timestamp_interval:based on ts_duration_of_tp and ts_num_tps from the timeseries table.
                Should be between 0 and 24.
    Output columns:
        * timestamp: dates based on the timepoints table
        * timeseries: the timeseries ID
        * time_row: the study period (planning_year) based on the timestamp
        * time_column: format: [r]mmdd, where r is an optional repetition number
            if there are multiple years of historical data in the timeseries
    """
    graph_timeseries_map = timepoints_df[["timestamp", "timeseries"]]
    graph_timeseries_map["time_row"] = graph_timeseries_map["timestamp"].str[:4]
    graph_timeseries_map["time_column"] = graph_timeseries_map["timestamp"].str[4:-2]

    return graph_timeseries_map


def loads_table(load_curves, timepoints_df):
    """
    Inputs:
        load_curves: from powergenome (one column per zone, one row per
            timepoint)
        timepoints_df: timepoints table previously prepared for switch,
            including timepoint_id column
    Output df
        loads: table of loads per load zone and timepoint
    Output columns
        * load_zone: the IPM regions
        * timepoint: from timepoints
        * zone_demand_mw: based on load_curves
    """

    # Assign timepoints and switch from one row per timepoint and one column per
    # zone to one row per timepoint-zone combination
    loads = load_curves.assign(TIMEPOINT=timepoints_df["timepoint_id"].values).melt(
        id_vars="TIMEPOINT",
        var_name="LOAD_ZONE",
        value_name="zone_demand_mw",
    )
    return loads[["LOAD_ZONE", "TIMEPOINT", "zone_demand_mw"]]


def variable_capacity_factors_table(all_gen_variability, all_gen, timepoints_df):
    """
    Inputs
        all_gen_variability: from powergenome (index=timepoint, columns=projects)
        all_gen: from powergenome
        timepoints_df: timepoints dataframe previously prepared for switch
    Output dataframe:
        GENERATION_PROJECT: from all_gen[Resource], filtered to include only
            variable projects PowerGenome's VRE == 1)
        TIMEPOINT: taken from index of all_gen_variability
        gen_max_capacity_factor: based on all_gen_variability
    """

    # switch from one row per timepoint and one column per project to one row per
    # timepoint-project combination and assign correct names
    variable_projects = all_gen.loc[
        (all_gen["VRE"] > 0) | (all_gen["FLEX"] > 0), "Resource"
    ]
    vcf = (
        all_gen_variability.loc[:, variable_projects]
        .assign(TIMEPOINT=timepoints_df["timepoint_id"].values)
        .melt(
            id_vars="TIMEPOINT",
            var_name="GENERATION_PROJECT",
            value_name="gen_max_capacity_factor",
        )
    )

    return vcf[["GENERATION_PROJECT", "TIMEPOINT", "gen_max_capacity_factor"]]


def load_zones_table(settings):
    regions = settings["model_regions"]
    load_zones = pd.DataFrame(
        {
            "LOAD_ZONE": regions,
            # not really needed, but may be used by UCSD/REAM to generate
            # short, unique project names for each zone
            "zone_dbid": range(1, len(regions) + 1),
        }
    )
    dist_loss = settings.get("avg_distribution_loss", None)
    if dist_loss is not None:
        load_zones["local_td_loss_rate"] = dist_loss
    return load_zones


def create_transm_line_col(lz1, lz2, zone_dict):
    t_line = zone_dict[lz1] + "-" + zone_dict[lz2]
    return t_line


def transmission_lines_table(
    line_loss, add_cap, tx_capex_mw_mile_dict, zone_dict, settings
):
    """
    Create transmission_lines table based on REAM Scenario 178
    Output Columns:
        TRANSMISSION_LINE: zone_dbid-zone_dbid for trans_lz1 and lz2
        trans_lz1: split PG transmission_path_name
        trans_lz2: split PG transmission_path_name
        trans_length_km: PG distance_mile * km_per_mile
        trans_efficiency: PG line_loss_percentage (1 - line_loss_percentage)
        existing_trans_cap: PG line_max_cap_flow. Take absolute value and take max of the two values
        trans_dbid: id number
        trans_derating_factor: assuming PG DerateCapRes_1 (0.95)
        trans_terrain_multiplier:
            trans_capital_cost_per_mw_km * trans_terrain_multiplier = the average of the two regions
            ('transmission_investment_cost')['tx']['capex_mw_mile'])
        trans_new_build_allowed: how to determine what is allowed. Assume all 1s to start
    """
    transmission_df = line_loss[
        [
            "Network_Lines",
            "transmission_path_name",
            "distance_mile",
            "Line_Loss_Percentage",
        ]
    ]

    # split to get trans_lz1 and trans_lz2
    split_path_name = transmission_df["transmission_path_name"].str.split(
        "_to_", expand=True
    )
    transmission_df = transmission_df.join(split_path_name)

    # convert miles to km for trans_length_km
    transmission_df["trans_length_km"] = transmission_df["distance_mile"] * km_per_mile

    # for trans_efficiency do 1 - line_loss_percentage
    transmission_df["trans_efficiency"] = transmission_df["Line_Loss_Percentage"].apply(
        lambda x: 1 - x
    )

    transmission_df = transmission_df.join(
        add_cap[["Line_Max_Flow_MW", "Line_Min_Flow_MW", "DerateCapRes_1"]]
    )

    # want the max value so take abosolute of line_min_flow_mw (has negatives) and then take max
    transmission_df["line_min_abs"] = transmission_df["Line_Min_Flow_MW"].abs()
    transmission_df["existing_trans_cap"] = transmission_df[
        ["Line_Max_Flow_MW", "line_min_abs"]
    ].max(axis=1)

    # get rid of columns
    transm_final = transmission_df.drop(
        [
            "transmission_path_name",
            "distance_mile",
            "Line_Loss_Percentage",
            "Line_Max_Flow_MW",
            "Line_Min_Flow_MW",
            "line_min_abs",
        ],
        axis=1,
    )

    transm_final = transm_final.rename(
        columns={
            "Network_Lines": "trans_dbid",
            0: "trans_lz1",
            1: "trans_lz2",
            "DerateCapRes_1": "trans_derating_factor",
        }
    )

    transm_final["tz1_dbid"] = transm_final["trans_lz1"].apply(lambda x: zone_dict[x])
    transm_final["tz2_dbid"] = transm_final["trans_lz2"].apply(lambda x: zone_dict[x])
    transm_final["TRANSMISSION_LINE"] = (
        transm_final["tz1_dbid"].astype(str)
        + "-"
        + transm_final["tz2_dbid"].astype(str)
    )

    # use average of transmission cost from the two zones and convert to cost per MW-km
    transm_final["cap_cost_per_mw_km"] = (
        0.5
        * (
            transm_final["trans_lz1"].map(tx_capex_mw_mile_dict)
            + transm_final["trans_lz2"].map(tx_capex_mw_mile_dict)
        )
        / km_per_mile  # ($/mi) / (km/mi) = $/km
    )
    # benchmark value for transmission capital cost; each line will end up with
    # trans_capital_cost_per_mw_km * trans_terrain_multiplier = cap_cost_per_mw_km
    trans_capital_cost_per_mw_km = transm_final["cap_cost_per_mw_km"].min()
    if trans_capital_cost_per_mw_km == 0:
        trans_capital_cost_per_mw_km = 1  # avoid division by zero below

    transm_final["trans_terrain_multiplier"] = (
        transm_final["cap_cost_per_mw_km"] / trans_capital_cost_per_mw_km
    )

    # TODO: set trans_new_build_allowed if available in PowerGenome.
    # For now, we allow expansion on all lines (which is the default
    # in Switch) and use trans_path_expansion_limit.csv (in pg_to_switch.py)
    # to restrict transmission construction if needed.
    transm_final["trans_new_build_allowed"] = 1

    # sort columns
    transm_final = transm_final[
        [
            "TRANSMISSION_LINE",
            "trans_lz1",
            "trans_lz2",
            "trans_length_km",
            "trans_efficiency",
            "existing_trans_cap",
            "trans_dbid",
            "trans_derating_factor",
            "trans_terrain_multiplier",
        ]
    ]
    return transm_final, trans_capital_cost_per_mw_km


def tx_cost_transform(tx_cost_df):
    tx_cost_df["cost_per_mw-km"] = (
        tx_cost_df["total_interconnect_cost_mw"] / tx_cost_df["total_mw-km_per_mw"]
    )
    trans_capital_cost_per_mw_km = tx_cost_df["cost_per_mw-km"].min()
    if trans_capital_cost_per_mw_km == 0:
        trans_capital_cost_per_mw_km = 1  # avoid division by zero later
    tx_cost_df["trans_terrain_multiplier"] = (
        tx_cost_df["cost_per_mw-km"] / trans_capital_cost_per_mw_km
    )
    tx_cost_df["trans_efficiency"] = 1 - tx_cost_df["total_line_loss_frac"]
    tx_cost_df["trans_length_km"] = tx_cost_df["total_mw-km_per_mw"]
    tx_cost_df["trans_new_build_allowed"] = 1
    tx_cost_df["existing_trans_cap"] = tx_cost_df["Line_Max_Flow_MW"]
    return tx_cost_df, trans_capital_cost_per_mw_km


def balancing_areas(
    pudl_engine,
    regions,
    all_gen,
    quickstart_res_load_frac,
    quickstart_res_wind_frac,
    quickstart_res_solar_frac,
    spinning_res_load_frac,
    spinning_res_wind_frac,
    spinning_res_solar_frac,
):
    """
    Function to create balancing_areas and zone_balancing_area tables
    Input:
        1) pudl_engine from init_pudl_connection
        2) IPM regions from settings.get('model_regions')
        3) all_gen pandas dataframe from gc.create_all_generators()
        4) quickstart_res_load_frac, quickstart_res_wind_frac, quickstart_res_solar_frac,
            spinning_res_load_frac, spinning_res_wind_frac, and spinning_res_solar_frac:
            --> set these equal to values based on REAM
    Output:
        BALANCING_AREAS
            * BALANCING_AREAS: based on balancing authority from pudl and connecting that to all_gen using plant_id_eia
            * other columns based on REAM Scenario 178
        ZONE_BALANCING_AREAS
            * Load_zone: IPM region
            * balancing_area
    """
    import pudl

    if pudl.__version__ <= "0.6.0":
        # get table from PUDL that has  balancing_authority_code_eia
        plants_entity_eia = pd.read_sql_table("plants_entity_eia", pudl_engine)
    else:
        plants_eia = pd.read_sql_table(
            "plants_eia860",
            pudl_engine,
            parse_dates=["report_date"],
            columns=["report_date", "plant_id_eia", "balancing_authority_code_eia"],
        )
        plants_entity_eia = plants_eia.sort_values(
            "report_date", ascending=False
        ).drop_duplicates(
            # take the latest reported row for this plant; this gets the latest
            # balancing_authority_code_eia if a plant has moved and also avoids
            # problems with unreported balancing_authority_code_eia's in some
            # earlier years.
            subset=["plant_id_eia"],
            keep="first",
        )
    # dataframe with only balancing_authority_code_eia and plant_id_eia
    plants_entity_eia = plants_entity_eia[
        ["balancing_authority_code_eia", "plant_id_eia"]
    ]
    # create a dictionary that has plant_id_eia as key and the balancing authority as value
    plants_entity_eia_dict = plants_entity_eia.set_index("plant_id_eia").T.to_dict(
        "list"
    )

    plant_region_df = all_gen.copy()
    plant_region_df = plant_region_df[["plant_id_eia", "region"]]

    # get rid of NAs
    plant_region_df = plant_region_df[plant_region_df["plant_id_eia"].notna()]

    """
    BALANCING_AREAS:
    take the plant_id_eia column from all_gen input, and return the balancing authority using
        the PUDL plants_entity_eia dictionary

    """

    # define function to get balancing_authority_code_eia from plant_id_eia
    def id_eia_to_bal_auth(plant_id_eia, plants_entity_eia_dict):
        if plant_id_eia in plants_entity_eia_dict.keys():
            return plants_entity_eia_dict[plant_id_eia][
                0
            ]  # get balancing_area from [balancing_area]
        else:
            return "-"

    # return balancing_authority_code_eia from PUDL table based on plant_id_eia
    plant_region_df["balancing_authority_code_eia"] = plant_region_df[
        "plant_id_eia"
    ].apply(lambda x: id_eia_to_bal_auth(x, plants_entity_eia_dict))

    # create output table
    balancing_areas = plant_region_df["balancing_authority_code_eia"].unique()
    BALANCING_AREAS = pd.DataFrame(balancing_areas, columns=["BALANCING_AREAS"])
    BALANCING_AREAS["quickstart_res_load_frac"] = quickstart_res_load_frac
    BALANCING_AREAS["quickstart_res_wind_frac"] = quickstart_res_wind_frac
    BALANCING_AREAS["quickstart_res_solar_frac"] = quickstart_res_solar_frac
    BALANCING_AREAS["spinning_res_load_frac"] = spinning_res_load_frac
    BALANCING_AREAS["spinning_res_wind_frac"] = spinning_res_wind_frac
    BALANCING_AREAS["spinning_res_solar_frac"] = spinning_res_solar_frac

    """
    ZONE_BALANCING_AREAS table:
        for each of the regions, find the most common balancing_authority to create table
    """

    zone_b_a_list = list()
    for region in regions:
        region_df = plant_region_df.loc[plant_region_df["region"] == region]
        # take the most common balancing authority (assumption)
        bal_aut = mode(region_df["balancing_authority_code_eia"].to_list())
        zone_b_a_list.append([region, bal_aut])
    zone_b_a_list.append(["_ALL_ZONES", "."])  # Last line in the REAM inputs
    ZONE_BALANCING_AREAS = pd.DataFrame(
        zone_b_a_list, columns=["LOAD_ZONE", "balancing_area"]
    )

    return BALANCING_AREAS, ZONE_BALANCING_AREAS
