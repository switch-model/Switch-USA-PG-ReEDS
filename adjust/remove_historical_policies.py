"""
Create alternative versions of
{rps,min_cap,max_cap}_{requirements,generators}.csv files that drop
CES/RPS/min-cap/max-cap policies for historical years, currently identified as
2025 or earlier. For single-period models, the same effect can be achieved by
omitting the min_capacity_constraint, max_capacity_constraint and rps_regional
modules. So this is mainly useful for mixed models with some historical and some
future periods. The downside of either of these approaches is that they will
mask the effect of RPS on prices, e.g., sometimes running at negative marginal
cost to meet RPS (but those seem unlikely).

Note: this does not remove carbon policies from historical years, because
they currently all have escape prices, so they will not cause infeasibility.
"""

import sys
from pathlib import Path

import pandas as pd

in_dir = Path(sys.argv[1])
last_hist_year = 2025

for pol in ["min_cap", "max_cap", "rps"]:
    rqmt_file = in_dir / f"{pol}_requirements.csv"
    gen_file = in_dir / f"{pol}_generators.csv"

    rqmts = pd.read_csv(rqmt_file, na_values=".")
    if (rqmts["PERIOD"] <= last_hist_year).any():
        rqmts = rqmts[rqmts["PERIOD"] > last_hist_year]
        rqmts.to_csv(rqmt_file, na_rep=".", index=False)
        print(f"Removed {pol} requirements from {rqmt_file} for historical years.")

        progs = rqmts.iloc[:, 0].unique()
        gens = pd.read_csv(gen_file, na_values=".")
        old_count = len(gens)
        gens = gens[gens.iloc[:, 0].isin(progs)]
        if len(gens) < old_count:
            gens.to_csv(gen_file, na_rep=".", index=False)
            print(f"Updated {gen_file} to match {rqmt_file}.")
