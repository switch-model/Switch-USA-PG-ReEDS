"""
This script copies two Switch input directories to a temporary folder,
standardizing them on the way, then uses the opendiff command to compare them.
This can be useful for seeing whether two versions of the data pipeline produce
the same results, ignoring insignificant differences.

Standardization consists of rounding all numerical values to 10 significant
figures, converting all-int columns from float to int, and sorting rows in each
file in ascending order across all columns.

usage: python compare_inputs.py old_dir new_dir
"""

import sys, tempfile, os, shutil, subprocess
from pathlib import Path
import pandas as pd, numpy as np


def round(x, n_figs):
    """
    Round series or dataframe to specified number of significant figures;
    this uses vectorized calculations so it is fairly fast.
    Note: numbers smaller than 1e-200 will be rounded to zero.
    """
    power = 10 ** (np.floor(np.log10(np.abs(x).clip(1e-200))))
    rounded = np.round(x / power, n_figs - 1) * power
    # also round numbers that are very close to zero to zero; useful when
    # comparing effectively zero results in outputs files
    rounded[rounded.abs() < 10**-n_figs] = 0
    return rounded


in1 = sys.argv[1]
in2 = sys.argv[2]
try:
    # trim identical parts from the start and end of the paths
    pp1 = Path(in1).parts
    pp2 = Path(in2).parts
    for start, (p1, p2) in enumerate(zip(pp1, pp2)):
        if p1 != p2:
            break
    for end, (p1, p2) in enumerate(zip(reversed(pp1), reversed(pp2))):
        if p1 != p2:
            break
    out1 = "_".join(pp1[start : len(pp1) - end])
    out2 = "_".join(pp2[start : len(pp2) - end])
except:
    # flag to use dummy names
    out1 = out2 = "bad"

if out1 == out2 or out1 == "" or out2 == "":
    out1 = in1.replace("\\", "-").replace("/", "-")
    out2 = in2.replace("\\", "-").replace("/", "-")

with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:

    for orig, out in [(in1, out1), (in2, out2)]:
        outdir = Path(td) / out
        outdir.mkdir()

        for root, dirs, files in os.walk(orig):
            for name in dirs:
                os.mkdir(os.path.join(outdir, name))
            for name in files:
                old_file = os.path.join(root, name)
                new_file = os.path.join(outdir, name)
                # skip big files (10+ MB)
                if os.stat(old_file).st_size > 1e7:
                    print(f"skipping {old_file} (large)")
                    continue
                print(f"copying {old_file}")
                if name.endswith(".csv"):
                    df = pd.read_csv(old_file, na_values=".", keep_default_na=False)
                    # patch some missing values to simplify comparisons
                    if name == "gen_info.csv":
                        df.loc[
                            df["gen_max_age"].isna()
                            & (df["gen_tech"] == "Battery_*_Moderate"),
                            "gen_max_age",
                        ] = 15
                        df.loc[
                            df["gen_max_age"].isna()
                            & (df["gen_tech"] == "distributed_generation"),
                            "gen_max_age",
                        ] = 100
                    df = df.sort_values(axis=0, by=df.columns.to_list())
                    # convert float columns to int if possible
                    floats = df.columns[df.dtypes == "float64"]
                    for c in floats:
                        # if (df[c] % 1 == 0).all():
                        #     df[c] = df[c].astype(int)
                        try:
                            df[c] = df[c].astype("Int64")  # can keep NaNs
                        except TypeError:
                            # non-convertible, e.g., has non-int or inf values
                            pass
                    # Some of the original input data are rounded to 10 digits
                    # (maybe by processing with R?); output data may vary by
                    # 1e-6 or so, so we round to that many digits of precision.
                    # We also use a float_format that converts floats to ints
                    # where possible and drops artifacts like trailing zeroes
                    # and small values
                    floats = df.columns[df.dtypes == "float64"]
                    df[floats] = round(df[floats], 5)
                    df.to_csv(new_file, index=False, na_rep=".", float_format="%.15g")
                else:
                    shutil.copy(old_file, new_file)

    subprocess.run(["opendiff", Path(td) / out1, Path(td) / out2])

    input("Press Enter to finish and delete the temporary files...")
