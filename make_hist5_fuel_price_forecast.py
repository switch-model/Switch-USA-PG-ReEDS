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

base_seds_url = "https://api.eia.gov/v2/seds/data/"

series_fuels = {
    "CLEID": "coal",  # coal price, electric power sector
    "NGEID": "naturalgas",  # gas price, electric power sector
}
print(f"Retrieving fuel prices for {start_year}-{end_year} from {base_seds_url}")

params = [
    ("api_key", eia_api_key),
    ("frequency", "annual"),
    ("data[]", "value"),
    ("start", str(start_year)),
    ("end", str(end_year)),
    ("sort[0][column]", "period"),
    ("sort[0][direction]", "asc"),
    ("length", "5000"),
] + [("facets[seriesId][]", s) for s in series_fuels.keys()]

resp = requests.get(base_seds_url, params=params)
resp.raise_for_status()

rows = resp.json()["response"]["data"]
state_prices = pd.DataFrame(rows)

# convert strings to numbers
state_prices["period"] = state_prices["period"].astype(int)
state_prices["value"] = state_prices["value"].astype(float)
# pick the right columns, rename some, drop US total and missing values
state_prices = (
    state_prices.loc[:, ["period", "stateId", "seriesId", "value"]]
    .query("stateId != 'US' and value > 0")
    .rename(columns={"period": "year", "stateId": "st"})
)
# lookup fuel name
state_prices["fuel"] = state_prices["seriesId"].map(series_fuels)

# adjust for inflation
cpi = get_cpi_data(start_year, end_year)
inflator = (
    cpi.query("year == @base_year")["value"].iloc[0] / cpi.set_index("year")["value"]
)
state_prices["price_real"] = state_prices["value"] * state_prices["year"].map(inflator)

# aggregate across years
state_fcst = state_prices.groupby(["st", "fuel"])["price_real"].mean().reset_index()

# for any states missing prices (defunct or absent plants),
# fill in with arbitrary 2x the national average (to avoid looking
# like they could be restarted cheaply)
fill_fcst = state_fcst.groupby("fuel")["price_real"].mean() * 2
all_st_fuels = pd.MultiIndex.from_product(
    [state_fcst["st"].unique(), state_fcst["fuel"].unique()], names=["st", "fuel"]
).to_frame(index=False)
state_fcst = all_st_fuels.merge(state_fcst, on=["st", "fuel"], how="left")
state_fcst["price_real"] = state_fcst["price_real"].fillna(
    state_fcst["fuel"].map(fill_fcst)
)

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
#       hist5:
#         # ... leave other user stuff here
#         user_fuel_usd_year:
#           coal: 2024
#           naturalgas: 2024
#         user_fuel_price:
#           naturalgas:
#             p1: 3.23
#             p2: 3.32
#             ...
#           coal:
#             p1: 3.21
#             p2: 3.21
#             ...

print(f"Adding `hist5` forecast to PowerGenome settings")
ss = read_yaml(yaml_files["scenario_settings"])

yaml_root = [
    "settings_management",
    "all_years",
    "fuel_price_forecast",
    "hist5",
]

# build dict of user fuel prices
d = {}
for r in zone_fcst.itertuples():
    d.setdefault(r.fuel, {})[r.region] = r.price_real

add_yaml_key(ss, yaml_root + ["user_fuel_usd_year"], {f: base_year for f in d.keys()})
# remove any previous user_fuel_price entries, but leave other
# elements under hist5 alone, in case there are user settings there
delete_yaml_keys(ss, yaml_root + ["user_fuel_price"])
add_yaml_key(ss, yaml_root + ["user_fuel_price"], d)
write_yaml(ss, yaml_files["scenario_settings"])
