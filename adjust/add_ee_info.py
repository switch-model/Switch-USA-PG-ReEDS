"""
incorporate EE savings
 - even if it's an incremental 0.5-1% energy
savings each year, that's a 2.5-5% reduction in overall load at fairly low
cost (we use 3.75%, based on 0.75%/year for 5 years)
([ACEEE](https://www.aceee.org/sites/default/files/pdfs/cost_of_saving_electricity_final_6-22-21.pdf)
estimated $24/MWh in 2018 -- around $31 now)?

"""

import sys
from pathlib import Path

import pandas as pd

ee_annual_cost_per_mwh = 31
ee_load_reduction_frac = 0.0375

# in_dir = Path("switch/in/2030/s40x1")
in_dir = Path(sys.argv[1])


def read_csv(file):
    return pd.read_csv(in_dir / file, na_values=".")


def to_csv(df, file):
    path = in_dir / file
    df.to_csv(path, na_rep=".", index=False)
    print(f"Created {path}.")


# hourly data (also works for switch_model.balancing.demand_response.simple)
ee_data = read_csv("loads.csv")
ee_data["ee_load_reduction"] = ee_load_reduction_frac * ee_data["zone_demand_mw"]
# lookup timepoint info for later
tp = (
    read_csv("timepoints.csv")
    .merge(read_csv("timeseries.csv"), on="timeseries")
    .merge(read_csv("periods.csv"), left_on="ts_period", right_on="INVESTMENT_PERIOD")
    .eval(
        "tp_weight=ts_duration_of_tp * ts_scale_to_period / (period_start - period_end + 1)"
    )
    .set_index("timepoint_id")
)
ee_data["PERIOD"] = ee_data["TIMEPOINT"].map(tp["ts_period"])
ee_data["tp_weight"] = ee_data["TIMEPOINT"].map(tp["tp_weight"])

# cost per year to deploy the full amount of DR available
# (used by study_modules.demand_response_investment)
ee_cost = (
    (ee_data["ee_load_reduction"] * ee_data["tp_weight"])
    .groupby([ee_data["LOAD_ZONE"], ee_data["PERIOD"]])
    .sum()
    .mul(ee_annual_cost_per_mwh)
    .reset_index()
    .rename(columns={0: "ee_annual_cost"})
)

ee_data = ee_data.drop(columns=["zone_demand_mw", "PERIOD", "tp_weight"])
to_csv(ee_data, "ee_data.csv")
to_csv(ee_cost, "ee_annual_cost.csv")
