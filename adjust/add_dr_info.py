"""
Define a 3% shiftable load for the cases that use the
study_modules.demand_response_investment module. This is more conservative than the [DOE liftoff
report](https://www.smartenergydecisions.com/wp-content/uploads/2025/04/liftoff_doe_virtualpowerplants2025update.pdf),
which estimated 80-160 GW of VPP potential (we have an 800 GW system
coincident peak in 2030). DOE estimated VPP costs at $43/kW-yr based on some Brattle work.
"""

import sys
from pathlib import Path

import pandas as pd

annual_cost_per_mw = 43000
shift_down_limit = 0.03
shift_up_limit = 0.24

# in_dir = Path("switch/in/2030/s40x1")
in_dir = Path(sys.argv[1])


def read_csv(file):
    return pd.read_csv(in_dir / file, na_values=".")


def to_csv(df, file):
    path = in_dir / file
    df.to_csv(path, na_rep=".", index=False)
    print(f"Created {path}.")


# hourly data (also works for switch_model.balancing.demand_response.simple)
dr_data = read_csv("loads.csv")
dr_data["dr_shift_down_limit"] = shift_down_limit * dr_data["zone_demand_mw"]
dr_data["dr_shift_up_limit"] = shift_up_limit * dr_data["zone_demand_mw"]
# lookup periods per timepoint for later
tp_period = (
    read_csv("timepoints.csv")
    .merge(read_csv("timeseries.csv"), on="timeseries")
    .set_index("timepoint_id")["ts_period"]
)
dr_data["PERIOD"] = dr_data["TIMEPOINT"].map(tp_period)

# cost per year to deploy the full amount of DR available
# (used by study_modules.demand_response_investment)
dr_cost = (
    dr_data.groupby(["LOAD_ZONE", "PERIOD"])["dr_shift_down_limit"]
    .max()
    .mul(annual_cost_per_mw)
    .reset_index()
    .rename(columns={"dr_shift_down_limit": "dr_annual_cost"})
)

del dr_data["zone_demand_mw"], dr_data["PERIOD"]
to_csv(dr_data, "dr_data.csv")
to_csv(dr_cost, "dr_annual_cost.csv")
