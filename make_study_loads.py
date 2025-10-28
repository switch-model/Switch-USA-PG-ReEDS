"""
Create baseline user load profiles for all future years using PowerGenome ReEDS
data for one specific historical year. Then create demand response ("flexible
load") shapes to represent load growth beyond that and international
imports/exports.

This grows 2023 PowerGenome loads to future years, using ICF growth rates
(growth_rates/zone_growth.csv), then adds the difference between that and the
existing PowerGenome loads as "load_growth". (This gives a fairly good match
to EIA's reports of 2023 and 2024 US loads.)

It also adds net exports as a "us_exports" flexible load (which should be
treated as inflexible in PowerGenome). This applies the month-hour average for a
recent period (currently 2024), since time-synced values aren't available for
the historical weather period (EIA 930 covers 2015-present, but ReEDS historical
loads and weather are for 2006-13). This may also better capture changes in
import/export behavior since the historical weather years.
"""

# %%#################
# Setup
####################
from pathlib import Path
import json
import pandas as pd
from powergenome.util import load_settings
from powergenome.generators import GeneratorClusters
from powergenome.util import (
    init_pudl_connection,
    load_settings,
)
from pg_to_switch import short_fn

# TODO: get from argv
settings_dir = "pg/settings"
# zonal annual growth rates; created by growth_rates/retrieve_icf_growth.py
# note: we could use PowerGenome's alt_growth_rate setting instead of adding
# a tranche of flexible load, but this lets us make it interruptible.
growth_file = "growth_rates/zone_growth.csv"
reeds_load_table = "load_curves_nrel_reeds"
base_year = 2023
start_year = 2023
end_year = 2030
# baseline loads will be stored here; pg/settings/demand.yml/regional_load_fn
# should point to this file
user_load_file = f"reeds_{base_year}_loads.csv.zip"
# load growth and exports will be stored here; pg/settings/flexible_load.yml/demand_response_fn
# should point to this file
demand_response_file = "load_adjustments.csv.zip"

# resource names; must match keys in pg/settings/flexible_load.yml/flexible_demand_resources
load_growth_resource_name = "load_growth"
exports_resource_name = "us_exports"
export_averaging_years = [2024]
# file within settings["RESOURCE_GROUPS"] with info on virtual generators
# representing imports (path and file will be created if not present)
imports_json = "imports/imports_group.json"

print(f"Reading settings from {settings_dir}")

settings = load_settings(settings_dir)
pudl_engine, pudl_out, pg_engine = init_pudl_connection(
    freq="AS",
    start_year=min(settings.get("eia_data_years")),
    end_year=max(settings.get("eia_data_years")),
    pudl_db=settings.get("PUDL_DB"),
    pg_db=settings.get("PG_DB"),
)

load_file_path = Path(settings["input_folder"]) / user_load_file
dr_file_path = Path(settings["input_folder"]) / demand_response_file


# TODO: read table(s) from pudl_engine instead of directly from latest PUDL
# (for replicability and consistency with other inputs); to do this, we will
# need to include core_eia930__hourly_interchange table in make_retro_pudl_data.py
# and decide whether to use the pre-2023-12 or post-2023-12 schema for it (and here).
def read_pudl(tbl):
    url = f"https://s3.us-west-2.amazonaws.com/pudl.catalyst.coop/nightly/{tbl}.parquet"
    print(f"Reading {url}")
    return pd.read_parquet(url)


# %%#################
# Generate user load profiles (only for one model year, will be reused)
####################
print(
    f"Reading {base_year} ReEDS loads from table {reeds_load_table} via "
    f"{pg_engine.url}"
)
base = pd.read_sql(
    f"select * from {reeds_load_table} where year = {base_year}",
    con=pg_engine,
).rename(columns={"year": "base_year"})

print(f"Saving {base_year} loads for {start_year}-{end_year} in {load_file_path}")
base_wide = (
    pd.concat(
        (base.assign(model_year=y) for y in range(start_year, end_year + 1)),
        ignore_index=True,
    )
    .assign(scenario="base")
    .pivot(
        columns=["model_year", "scenario", "region"],
        index="time_index",
        values="load_mw",
    )
    .sort_index(axis=0)
    .astype(int)
)
base_wide.to_csv(load_file_path, index=False)
del base_wide

# %%#################
# Calculate load growth each year (will be added as flexible load)
####################
print("Calculating base year stats")
base_stats = (
    base.groupby("region")
    .agg(avg_base=("load_mw", "mean"), peak_base=("load_mw", "max"))
    .reset_index()
)

# find the target growth rates, then apply those to get the target avg and peak
# load levels
print("Calculating model year target stats")
target_rates = pd.read_csv(growth_file).rename(columns={"load_zone": "region"})
for s in ["avg", "peak"]:
    # remove any negative growth
    target_rates[f"{s}_growth"] = target_rates[f"{s}_growth"].clip(0, None)
target_stats = base_stats.merge(target_rates)
# spread to all possible model years
target_stats = pd.concat(
    [target_stats.assign(year=y) for y in range(start_year, end_year + 1)],
    ignore_index=True,
)
# set target avg & peak MW using exponential growth from base_year
for s in ["avg", "peak"]:
    target_stats[f"{s}_targ"] = target_stats[f"{s}_base"] * (
        1 + target_stats[f"{s}_growth"]
    ) ** (target_stats["year"] - base_year)

# Find the scale and offset to add to the base load levels to get the target
# growth levels
# fraction f and base b are found by solving this equation:
# ab * (1 + f) + b = at
# pb * (1 + f) + b = pt
# => pt - at = (pb - ab) * (1 + f)
# => f = (pt - at) / (pb - ab) - 1
# => b = at - ab * (1 + f)
target_stats["fraction"] = target_stats.eval(
    "(peak_targ - avg_targ) / (peak_base - avg_base) - 1"
)
target_stats["base"] = target_stats.eval("avg_targ - avg_base * (1 + fraction)")

# Apply the base and fraction to calculate the incremental load through each year
# This will be treated as "flexible load", possibly interruptible in some scenarios
print(f"Calculating growth from {base_year} for {start_year}-{end_year}.")
growth = base.merge(target_stats[["region", "year", "base", "fraction"]], on="region")
growth["growth_mw"] = growth["load_mw"] * growth["fraction"] + growth["base"]
# remove a few cases with shrinking loads (growth in peak but not mean);
# may end up missing the mean target slightly
growth["growth_mw"] = growth["growth_mw"].clip(0, None)
# check the shape overall
# growth.query('year == 2030 & weather_year == 2013').eval('hour_of_year = time_index % 8760').groupby('hour_of_year')['growth_mw'].sum().plot(ylim=(0, None))

# convert to correct form for PowerGenome demand_response_fn:
# csv file with hourly profiles for demand response resources in each
# region/year/scenario. The top four rows are
#   1) the name of the DR resource (in settings['flexible_demand_resources'][2030].keys()),
#   2) the model year,
#   3) the scenario name (settings['demand_response'])
#   4) the model region from `model_regions`
growth["resource_name"] = load_growth_resource_name
growth["scenario"] = settings["demand_response"]
growth["growth_mw"] = growth["growth_mw"].round(3)  # don't need more than kW resolution

growth_wide = growth.pivot(
    index="time_index",
    columns=["resource_name", "year", "scenario", "region"],
    values="growth_mw",
).sort_index(axis=0)
del growth
# Don't write now; will be stored later
# print(
#     f"Saving hourly flexible load profiles for {start_year}-{end_year} in {dr_file_path}."
# )
# growth_wide.to_csv(dr_file_path, index=False)
# print(f"Finished writing {dr_file_path}.")

# %%############
# Calculate net exports for each zone by month and hour, then use those to define
# us_exports "flexible" load (for exports) and virtual generators (for imports)
###############
print("Calculating net US exports for each load zone")

ba_trade = read_pudl("core_eia930__hourly_interchange")
ba = read_pudl("core_eia__codes_balancing_authorities").set_index("code")

# simplify column names and get neighbor region name
# note: positive interchange indicates exports (https://www.eia.gov/electricity/gridmonitor/about)
ba_trade = ba_trade.rename(
    columns={
        "interchange_reported_mwh": "exports",
        "balancing_authority_code_eia": "ba_code",
        "balancing_authority_code_adjacent_eia": "neighbor_code",
    }
)
ba_trade["neighbor_region"] = ba_trade["neighbor_code"].map(
    ba["balancing_authority_region_name_eia"]
)

pair_trade = (
    ba_trade.query(f"neighbor_region.isin({'Canada', 'Mexico'})")
    .groupby(["datetime_utc", "neighbor_code", "ba_code"])["exports"]
    .sum()
    .reset_index()
)

# Get averages by month of year and hour of day, in model time zone
print("Calculating average US exports for each month-hour combination")
pair_trade["datetime"] = pair_trade["datetime_utc"] + pd.Timedelta(
    hours=settings["utc_offset"]
)
pair_trade["month"] = pair_trade["datetime"].dt.month
pair_trade["hour"] = pair_trade["datetime"].dt.hour
avg = (
    pair_trade[pair_trade["datetime"].dt.year.isin(export_averaging_years)]
    .groupby(["neighbor_code", "ba_code", "month", "hour"])["exports"]
    .mean()
    .reset_index()
)

# apply shares of each external-internal BA pair to matching ReEDS regions
shares = pd.read_csv(Path(settings["input_folder"]) / "import_reeds_region_shares.csv")
avg = avg.merge(shares, on=["neighbor_code", "ba_code"], how="left")
assert (
    shares["share"].notna().all()
), "International interchange reported for unknown BA pairs."
avg["exports"] *= avg["share"]
avg = avg.groupby(["reeds_region", "month", "hour"])["exports"].sum().reset_index()

# for testing:
# dr_file_path = Path(settings["input_folder"]) / settings["demand_response_fn"]
# print(f"Reading previously stored flexible loads from {dr_file_path}")
# growth_wide = pd.read_csv(dr_file_path, header=[0, 1, 2, 3])

# make a time index the same length as other historical data (e.g., 7 sample
# years)
n_years = len(growth_wide) / 8760
assert n_years == int(n_years), "Loads are not an integer number of 8760-hour blocks"

time_index = pd.DataFrame(
    {
        # create a dummy datetime for any non-leap-year
        "datetime": pd.date_range(
            start="2025-01-01 00:00:00", periods=365 * 24, freq="H"
        )
    }
)
time_index["month"] = time_index["datetime"].dt.month
time_index["hour"] = time_index["datetime"].dt.hour
del time_index["datetime"]  # no longer needed, prevent confusion
time_index = pd.concat([time_index] * int(n_years), axis=0, ignore_index=True)
# assign time_index column matching growth table for reference later
time_index["time_index"] = growth_wide.index

# assign average loads along the whole time index
trade_long = time_index.merge(avg, on=["month", "hour"])[
    ["reeds_region", "time_index", "exports"]
]
assert (
    trade_long["exports"].notna().all
), "Unexpected nans found for exports, may be able to fill with 0"

# repeat for all possible model years and convert to wide format for powergenome
trade_wide = (
    pd.concat(
        (trade_long.assign(model_year=y) for y in range(start_year, end_year + 1)),
        ignore_index=True,
    )
    .assign(scenario="base", resource_name=exports_resource_name)
    .pivot(
        columns=["resource_name", "model_year", "scenario", "reeds_region"],
        index="time_index",
        values="exports",
    )
    .sort_index(axis=0)
    .astype(int)
)
# split into positive and negative versions, to save as extra loads and dummy generator
# profiles, respectively (similar to how ReEDS treats exports and imports)

exports_wide = trade_wide.clip(0, None)
imports_wide = (-trade_wide).clip(0, None)
# keep only columns with nonzero values
exports_wide = exports_wide.loc[:, (exports_wide != 0).any(axis=0)]
imports_wide = imports_wide.loc[:, (imports_wide != 0).any(axis=0)]

# Add exports to load
flex = pd.concat([growth_wide, exports_wide], axis=1)
print(
    f"Saving hourly load growth and export profiles for {start_year}-{end_year} in {dr_file_path}."
)
flex.to_csv(dr_file_path, index=False)
print(f"Finished writing {dr_file_path}.")

# Create virtual generator profiles for imports
print("Creating virtual generator profiles for US imports.")

# Use first year of imports, get peak production and normalize
first_year_index = imports_wide.columns[0][:3]
imports_wide = imports_wide.loc[:, first_year_index]
imports_capacity = imports_wide.max(axis=0)
imports_wide /= imports_capacity

# convert column names from region to dummy csa_id (int)
region_cpa = {k: str(i) for i, k in enumerate(imports_capacity.index)}
imports_wide = imports_wide.rename(columns=region_cpa)

# write to input files
imports_data_path = Path(settings["RESOURCE_GROUPS"]) / imports_json

# create imports json file if needed
if not imports_data_path.exists():
    imports_data_path.parent.mkdir(parents=True, exist_ok=True)
    imports_data = {
        "technology": "imports",
        "metadata": "imports_metadata.csv",
        "profiles": "imports_profiles.csv",
    }
    with open(imports_data_path, "w") as f:
        json.dump(imports_data, f, indent=4)

# get names of input files
with open(imports_data_path, "r") as f:
    imports_data = json.load(f)

profile_path = imports_data_path.parent / imports_data["profiles"]
imports_wide.to_csv(profile_path, index=False)
print(f"Saved {profile_path}")

metadata_path = imports_data_path.parent / imports_data["metadata"]
pd.DataFrame(
    {
        # all these columns seem to be needed for new VRE
        "region": imports_capacity.index,
        "id": imports_capacity.index,
        "cpa_id": imports_capacity.index.map(region_cpa),
        "mw": imports_capacity,
    }
).to_csv(metadata_path, index=False)
print(f"Saved {metadata_path}")

# %%
