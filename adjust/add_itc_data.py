import sys
from pathlib import Path
import pandas as pd
from powergenome.util import load_settings

in_dir = Path(sys.argv[1])
itc_file = in_dir / "itc.csv"

settings = load_settings("pg/settings")

pd.DataFrame.from_records(
    settings["itc"],
    columns=["ENERGY_SOURCE", "START_YEAR", "END_YEAR", "itc_rate", "itc_eligible_mw"],
).to_csv(itc_file, index=False, na_rep=".")
print(f"Created {itc_file}")
