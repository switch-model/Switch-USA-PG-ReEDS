"""
Add part-load heat rates to thermal power plants and reserve balancing areas
in the specified Switch model directory.

Part load heat rates: Assumes 20% fuel to run at 0 power, then linear up to
full-load heat rate. Represents each gen as if it were 1 MW.

gen_inc_heat_rates.csv
    project, power_start_mw, power_end_mw,
    incremental_heat_rate_mbtu_per_mwhr, fuel_use_rate_mmbtu_per_h

power_start_mw  power_end_mw  incremental_heat_rate_mbtu_per_mwhr  fuel_use_rate_mmbtu_per_h
min_load * 1    .             .                                    0.2 * full-load-heat-rate * 1 (0-load fuel) + 0.8 * full-load-heat-rate * min_load * 1 (incremental HR up to min load)
min_load * 1    1             0.8 * full-load-heat-rate            .

Balancing areas: sets load_zones.zone_balancing_area to matching `rto` from ReEDS
shapefile.



"""

import sys, os
from pathlib import Path

import pandas as pd

try:
    # prevent warnings when importing geopandas if PyGEOS and shapely are both available
    import shapely

    os.environ["USE_PYGEOS"] = "0"
except:
    pass
import geopandas as gpd

from powergenome.util import load_settings

settings = load_settings(path="pg/settings")


# sys.argv[1] = "switch/in/test/2030/s20_1"
in_dir = Path(sys.argv[1])

# zero-load fuel as a fraction of full-load fuel
zero_load_fuel = 0.2
# treat as 1 MW generator (will be scaled automatically)
gen_size = 1

gen_info = pd.read_csv(in_dir / "gen_info.csv", na_values=".").query(
    "gen_full_load_heat_rate.notna()"
)
# treat missing min load as 0
gen_info["gen_min_load_fraction"] = gen_info["gen_min_load_fraction"].fillna(0.0)

min_fuel = pd.DataFrame(
    {
        "GENERATION_PROJECT": gen_info["GENERATION_PROJECT"],
        "power_start_mw": gen_info["gen_min_load_fraction"] * gen_size,
        # zero-load fuel plus incremental fuel from zero up to min load
        "fuel_use_rate_mmbtu_per_h": gen_info.eval(
            "@zero_load_fuel * gen_full_load_heat_rate * @gen_size "
            "+ (1 - @zero_load_fuel) * gen_full_load_heat_rate * gen_min_load_fraction * @gen_size"
        ),
    }
)
ihr = pd.DataFrame(
    {
        "GENERATION_PROJECT": gen_info["GENERATION_PROJECT"],
        "power_start_mw": gen_info["gen_min_load_fraction"] * gen_size,
        "power_end_mw": 1 * gen_size,
        "incremental_heat_rate_mbtu_per_mwhr": (1 - zero_load_fuel)
        * gen_info["gen_full_load_heat_rate"],
    }
)

gen_ihr = pd.concat([min_fuel, ihr], axis=0).sort_values(
    ["GENERATION_PROJECT", "power_start_mw", "fuel_use_rate_mmbtu_per_h"]
)[
    [
        "GENERATION_PROJECT",
        "power_start_mw",
        "power_end_mw",
        "incremental_heat_rate_mbtu_per_mwhr",
        "fuel_use_rate_mmbtu_per_h",
    ]
]

gen_ihr.to_csv(in_dir / "gen_inc_heat_rates.csv", na_rep=".", index=False)
print(f"Created {in_dir / 'gen_inc_heat_rates.csv'}")

#######################
# add `rto` from ReEDS shapefile to load_zones.csv as `zone_balancing_area`
zone_info = gpd.read_file(
    Path(settings["input_folder"]) / settings["user_region_geodata_fn"]
)

lz_file = in_dir / "load_zones.csv"
load_zones = pd.read_csv(lz_file, na_values=".")
load_zones["zone_balancing_area"] = load_zones["LOAD_ZONE"].map(
    zone_info.set_index("region")["rto"]
)
load_zones.to_csv(lz_file, na_rep=".", index=False)
print(f"Added `zone_balancing_area` column to {lz_file}.")
