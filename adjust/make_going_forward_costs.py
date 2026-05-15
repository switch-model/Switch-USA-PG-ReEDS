import sys
from pathlib import Path

import pandas as pd


# clean up pandas floating point output
def clean_to_csv(self, *args, float_format="%.15g", **kwargs):
    pd_to_csv(self, *args, float_format=float_format, **kwargs)


pd_to_csv = pd.DataFrame.to_csv
pd.DataFrame.to_csv = clean_to_csv


in_dir = Path(sys.argv[1])
# in_dir = Path("switch/in/2030/ce")
first_future_year = 2026

gen_costs_file = in_dir / "gen_build_costs.csv"

gen_info = pd.read_csv(in_dir / "gen_info.csv", na_values=".")
gen_costs = pd.read_csv(gen_costs_file, na_values=".")

tech_crosswalk = pd.DataFrame(
    [
        ["Batteries", "Utility-Scale Battery Storage_Lithium Ion_Advanced"],
        [
            "Natural Gas Fired Combined Cycle",
            "NaturalGas_1-on-1 Combined Cycle (H-Frame)_Moderate",
        ],
        [
            "Natural Gas Fired Combustion Turbine",
            "NaturalGas_Combustion Turbine (F-Frame)_Moderate",
        ],
        ["Nuclear", "Nuclear_Nuclear - Large_Moderate"],
        ["Onshore Wind Turbine", "LandbasedWind_Class3_Conservative"],
        ["Solar Photovoltaic", "UtilityPV_Class1_Moderate"],
        ["Offshore Wind Turbine", "OffShoreWind_Class3_Conservative_fixed_1"],
    ],
    columns=["old_tech", "new_tech"],
)

cols = [
    "GENERATION_PROJECT",
    "gen_tech",
    "gen_load_zone",
    "gen_connect_cost_per_mw",
    "gen_amortization_period",
]
gen_crosswalk = (
    gen_info[cols]
    .merge(tech_crosswalk, left_on="gen_tech", right_on="old_tech")
    .merge(
        gen_info[cols],
        left_on=["new_tech", "gen_load_zone"],
        right_on=["gen_tech", "gen_load_zone"],
    )
    .rename(
        columns={
            "GENERATION_PROJECT_x": "old_gen_proj",
            "GENERATION_PROJECT_y": "new_gen_proj",
            "gen_connect_cost_per_mw_y": "gen_connect_cost_per_mw",
            "gen_amortization_period_y": "gen_amortization_period",
        }
    )
)[
    [
        "old_gen_proj",
        "new_gen_proj",
        "gen_connect_cost_per_mw",
        "gen_amortization_period",
    ]
]

# some old projects may have matched to two new projects if there are multiple
# clusters; any one should do OK

# TODO: average the gen_overnight_cost and gen_storage_overnight_cost for matching projects
# Apply the average connection cost
# Get mean connect cost and capex. Note: we don't apply the connect cost to gen_info
# to avoid having it be used for plants built before first_future_year.
# connect_cost_map = gen_crosswalk.groupby('old_gen_proj')['gen_connect_cost_per_mw'].mean()

# assumes there will only be one match for each new-build tech
cost_crosswalk = gen_crosswalk.merge(
    gen_costs, left_on="new_gen_proj", right_on="GENERATION_PROJECT"
)
assert (
    len(cost_crosswalk["BUILD_YEAR"].unique()) == 1
), "Found costs for multiple years."

# add the connect cost to the capex per MW
cost_crosswalk["gen_overnight_cost"] += cost_crosswalk[
    "gen_connect_cost_per_mw"
].fillna(0)

# use the simple average cost if there are multiple clusters
new_costs = cost_crosswalk.groupby("old_gen_proj")[
    ["gen_overnight_cost", "gen_storage_energy_overnight_cost"]
].mean()

for col in ["gen_overnight_cost", "gen_storage_energy_overnight_cost"]:
    update_mask = (gen_costs[col] == 0) & (gen_costs["BUILD_YEAR"] >= first_future_year)
    gen_costs.loc[update_mask, col] = (
        gen_costs.loc[update_mask, "GENERATION_PROJECT"]
        .map(new_costs[col])
        .fillna(gen_costs.loc[update_mask, col])
    )

unmatched = gen_costs.query(
    "BUILD_YEAR >= @first_future_year and gen_overnight_cost==0 and not GENERATION_PROJECT.str.contains('distributed') and not GENERATION_PROJECT.str.contains('load_growth') and not GENERATION_PROJECT.str.contains('imports')"
)
if not unmatched.empty:
    print(
        f"The following {len(unmatched)} generation projects do not "
        "have costs assigned because they do not have a matching new-build "
        "project definition:"
    )
    print(
        ", ".join(
            f"{r['GENERATION_PROJECT']} ({r['BUILD_YEAR']})"
            for i, r in unmatched.iterrows()
        )
    )
    print()

gen_costs.to_csv(gen_costs_file, na_rep=".", index=False)
print(
    f"Assigned new-build capital costs (including connection cost) in "
    f"{gen_costs_file} for pre-planned generators built in {first_future_year} "
    f"or later."
)

trans_file = in_dir / "trans_params.csv"
trans = pd.read_csv(trans_file, na_values=".")
trans["trans_capital_cost_per_mw_km"] = 0
trans.to_csv(trans_file, na_rep=".", index=False)
print(f"Set trans_capital_cost_per_mw_km in {trans_file}.")

# apply correct amortization period, since capital recovery will now be important
amort_map = gen_crosswalk.groupby("old_gen_proj")["gen_amortization_period"].mean()
for f in in_dir.glob("gen_info*.csv"):
    gi = pd.read_csv(f, na_values=".")
    gi["gen_amortization_period"] = gi["gen_amortization_period"].fillna(
        gi["GENERATION_PROJECT"].map(amort_map)
    )
    gi.to_csv(f, na_rep=".", index=False)
    print(f"Set gen_amortization_period in {f}.")
