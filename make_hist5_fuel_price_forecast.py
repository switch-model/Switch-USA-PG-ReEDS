"""
Retrieve coal and gas prices for electricity production for
each state from the EIA SEDS system using the EIA API, then
use these to define fuel price forecasts for all future years
via settings like this in scenario_management.yml

settings_management:
  all_years:
    fuel_price_forecast:
      aeo2025: ~   # use default
      hist5:
        user_fuel_usd_year:
          coal: 2024
          naturalgas: 2024
        user_fuel_price:
          naturalgas:
            p1: 3.23
            p2: 3.32
            ...
          coal:
            p1: 3.21
            p2: 3.21
            ...
"""

print("loading libraries")
import os
from pathlib import Path
import requests

import pandas as pd
import ruamel.yaml
from ruamel.yaml.comments import CommentedMap, CommentedSeq

try:
    # prevent warnings when importing geopandas if PyGEOS and shapely are both available
    import shapely

    os.environ["USE_PYGEOS"] = "0"
except:
    pass
import geopandas as gpd

from powergenome.util import load_settings
from powergenome.financials import get_cpi_data

from utilities import read_yaml, write_yaml, add_yaml_key, delete_yaml_keys

# date range to retrieve prices for
start_year = 2020
end_year = 2024

settings_path = Path("pg/settings")

yaml_files = {
    # key: file containing it
    "scenario_settings": settings_path
    / "scenario_management.yml",
}

print(f"loading PowerGenome settings from {settings_path}")
settings = load_settings(settings_path)
# year to inflate prices to
base_year = settings.get("target_usd_year") or end_year

print(f"checking EIA API key")
eia_api_key_file = Path(settings["input_folder"]) / "eia_api_key.txt"
if not eia_api_key_file.is_file():
    raise RuntimeError(
        f"No EIA API key found. Please register for an API key at "
        f"https://www.eia.gov/opendata/register.php then store the "
        "key in {eia_api_key_file}"
    )
eia_api_key = eia_api_key_file.read_text().strip()

################
# Retrieve EIA historical prices

# See SEDS query constructor at https://www.eia.gov/opendata/browser/seds to get example URL like
# https://api.eia.gov/v2/seds/data/?frequency=annual&data[0]=value&facets[seriesId][]=CLEID&facets[seriesId][]=NGEID&start=2020&end=2024&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000

# Also see EIA electric power operations (EPO) query builder at
# https://www.eia.gov/opendata/browser/electricity/electric-power-operational-data
# EIA electric power API generally has more recent data than SEDS (which has a ~18 mo lag)
# but it lacks nuclear, which is in SEDS. 2023 and 2024 have similar
# availability but state/sector/fuel coverage can be patchy, e.g.,
#              Electric Utility Electric Power
#                             | |
# subbituminous coal          + -
# coal, excluding waste coal  + -
# all coal products           + +  EP = SEDS
# petroleum coke              - -
# refined coal                - -
# natural gas                 + * TX, UT, not MO  EP =~ SEDS
# natural gas & other gases   + -
# distillate fuel oil         + -  EU = SEDS
# petroleum liquids           + * TX, MO, not UT
# petroleum                   + -
# residual fuel oil           - -
# nuclear                     - -

base_seds_url = "https://api.eia.gov/v2/seds/data/"
base_epod_url = (
    "https://api.eia.gov/v2/electricity/electric-power-operational-data/data/"
)

# from SEDS query constructor, SERIESID filtered to
# "price in the electric power sector"
seds_fuels = {
    "CLEID": "coal",
    "NGEID": "naturalgas",
    "DKEID": "distillate",
    "NUEGD": "uranium",
}
# from EPO query constructor, FUELTYPEID filtered to a fuel from
# the EPO table above
# "price in the electric power sector"
epod_fuels = {
    "COW": "coal",
    "NG": "naturalgas",
    "DFO": "distillate",
    "NUC": "uranium",
}
# possible sector listsings for power prices, in priority order
epod_sectors = {98: "Electric Power", 1: "Electric Utility"}

print(f"Retrieving fuel prices for {start_year}-{end_year} from {base_seds_url}")
seds_params = [
    ("api_key", eia_api_key),
    ("frequency", "annual"),
    ("data[]", "value"),
    ("start", str(start_year)),
    ("end", str(end_year)),
    ("sort[0][column]", "period"),
    ("sort[0][direction]", "asc"),
    ("length", "5000"),
] + [("facets[seriesId][]", s) for s in seds_fuels.keys()]

resp = requests.get(base_seds_url, params=seds_params)
resp.raise_for_status()

rows = resp.json()["response"]["data"]

seds_prices = (
    pd.DataFrame(rows)
    .rename(columns={"period": "year", "stateId": "st", "value": "price"})
    .eval("price = price.astype('float64')")
    .eval("year = year.astype('Int64')")
    .eval("fuel = seriesId.map(@seds_fuels)")
    .query("st != 'US' and price > 0")
    .loc[:, ["year", "st", "fuel", "price"]]
)

# initialize state_prices dataframe
state_prices = seds_prices.copy()

# fill in with electric power or electric utility data from
# EPO API where needed
print(
    f"Retrieving additional fuel prices for {start_year}-{end_year} from {base_epod_url}"
)
epod_params = (
    [
        ("api_key", eia_api_key),
        ("frequency", "annual"),
        ("data[0]", "cost-per-btu"),
        ("start", str(start_year)),
        ("end", str(end_year)),
        ("sort[0][column]", "period"),
        ("sort[0][direction]", "asc"),
        ("length", "5000"),
    ]
    + [("facets[sectorid][]", s) for s in epod_sectors.keys()]
    + [("facets[fueltypeid][]", f) for f in epod_fuels.keys()]
)

resp = requests.get(base_epod_url, params=epod_params)
resp.raise_for_status()

rows = resp.json()["response"]["data"]

# convert strings to numbers
# pick the right columns, rename some, drop US total and missing values
epod_prices = (
    pd.DataFrame(rows)
    .rename(columns={"period": "year", "location": "st", "cost-per-btu": "price"})
    .eval("price = price.astype('float64')")
    .eval("year = year.astype('Int64')")
    .eval("fuel = fueltypeid.map(@epod_fuels)")
    .eval("sector = sectorid.astype('Int64').map(@epod_sectors)")
    .query("st.isin(@state_prices['st']) and price > 0")
    .loc[:, ["year", "st", "sector", "fuel", "price"]]
)


def add_missing(df_to, df_from):
    key_cols = ["year", "st", "fuel"]
    to_add = (
        df_from.merge(
            df_to[key_cols],
            on=key_cols,
            how="left",
            indicator=True,
        )
        .query("_merge == 'left_only'")
        .drop(columns="_merge")
    )
    result = pd.concat([df_to, to_add], ignore_index=True)[df_to.columns]
    return result


# add EPO prices to state prices if needed, preferring Electric Power sector if
# available, otherwise Electric Utility sector

for s in epod_sectors.values():
    state_prices = add_missing(state_prices, epod_prices.query("sector == @s"))

print(
    f"Retrieved {len(seds_prices)} prices from SEDS and added {len(state_prices)-len(seds_prices)} from EPO."
)

# adjust for inflation
cpi = get_cpi_data(start_year, end_year)
inflator = (
    cpi.query("year == @base_year")["value"].iloc[0] / cpi.set_index("year")["value"]
)
state_prices["price_real"] = state_prices["price"] * state_prices["year"].map(inflator)

# make hist5 and peak forecasts
fcst_info = [("hist5", state_prices["year"].unique())] + [
    (f"y{y}", [y]) for y in range(start_year, end_year + 1)
]
fcst_dfs = []
for fcst_name, fcst_base_years in fcst_info:
    # aggregate across years
    fcst = (
        state_prices.query("year.isin(@fcst_base_years)")
        .groupby(["st", "fuel"])["price_real"]
        .mean()
        .reset_index()
    )
    # for any states missing prices (defunct or absent plants),
    # fill in with arbitrary 2x the national average (to avoid looking
    # like they could be restarted cheaply)
    fill_fcst = fcst.groupby("fuel")["price_real"].mean() * 2
    all_st_fuels = pd.MultiIndex.from_product(
        [fcst["st"].unique(), fcst["fuel"].unique()], names=["st", "fuel"]
    ).to_frame(index=False)
    fcst = all_st_fuels.merge(fcst, on=["st", "fuel"], how="left")
    fcst["price_real"] = fcst["price_real"].fillna(fcst["fuel"].map(fill_fcst))
    fcst["forecast_name"] = fcst_name
    fcst_dfs.append(fcst)

state_fcst = pd.concat(fcst_dfs)

# assign to regions
region_info = gpd.read_file(
    Path(settings["input_folder"]) / settings["user_region_geodata_fn"]
)
# convert to capitals to match other data
region_info["st"] = region_info["st"].str.upper()

zone_fcst = state_fcst.merge(region_info[["st", "region"]]).sort_values(
    ["region", "fuel"]
)

###
# define the `hist5:` entry like
# settings_management:
#   all_years:
#     fuel_price_forecast:
#       # ... leave other user stuff here
#       hist5:
#         user_fuel_usd_year:
#           coal: 2024
#           naturalgas: 2024
#           ...
#         user_fuel_price:
#           naturalgas:
#             p1: 3.23
#             p2: 3.32
#             ...
#           coal:
#             p1: 3.21
#             p2: 3.21
#             ...
#           ...
#       y2022:
#         user_fuel_usd_year:
#           coal: 2024
#           ...
#         user_fuel_price:
#           coal:
#             p1: 3.23
#             p2: 3.32
#             ...
#           ...
#         ...

# build dict of user fuel prices
d = {}
for r in zone_fcst[["forecast_name", "fuel"]].drop_duplicates().itertuples():
    d.setdefault(r.forecast_name, {}).setdefault("user_fuel_usd_year", {})[
        r.fuel
    ] = base_year
for r in zone_fcst.itertuples():
    d[r.forecast_name].setdefault("user_fuel_price", {}).setdefault(r.fuel, {})[
        r.region
    ] = r.price_real

fcst_names = [f"`{k}`" for k in d.keys()]
if len(fcst_names) == 0:
    names = "<no forecasts>"
elif len(fcst_names) == 1:
    names = fcst_names[0] + " forecast"
else:
    names = ", ".join(fcst_names[:-1]) + " and " + fcst_names[-1] + " forecasts"

print(f"Adding {names} to PowerGenome settings")
ss = read_yaml(yaml_files["scenario_settings"])

yaml_root = ["settings_management", "all_years", "fuel_price_forecast"]
for fcst_name, fcst in d.items():
    # remove any previous forecasts with these names, but leave other
    # forecasts alone
    yaml_path = yaml_root + [fcst_name]
    delete_yaml_keys(ss, yaml_path)
    add_yaml_key(ss, yaml_path, fcst)

write_yaml(ss, yaml_files["scenario_settings"])
