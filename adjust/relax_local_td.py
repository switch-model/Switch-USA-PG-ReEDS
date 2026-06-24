"""
Set existing_local_td to arbitrarily large value (10 TW) for all load zones.
This allows using the local_td module to calculate T&D losses without requiring
investment in new local T&D (which can also affect the feasibility iteration
unnecessarily).
"""

import sys
from pathlib import Path

import pandas as pd

# in_dir = Path("in/2030/s1x1")
in_dir = Path(sys.argv[1])


def read_csv(file):
    return pd.read_csv(in_dir / file, na_values=".")


def to_csv(df, file):
    path = in_dir / file
    df.to_csv(path, na_rep=".", index=False)
    print(f"Created {path}.")


lz = read_csv("load_zones.csv")
lz["existing_local_td"] = 10_000_000

to_csv(lz, "load_zones.csv")
