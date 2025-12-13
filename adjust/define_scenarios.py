"""
Create alternative versions of gen_info.csv that have various permutations of

- a clean power production tax (to drive more fossil investment)
- no retirement of existing generators (to sustain fossil infrastructure)
- flexibility for grown loads

These will be:

- gen_info.high_fossil.csv  # used for high_fossil.build case
- gen_info.no_retire.csv    # used for high_fossil (cost evaluation) case
- gen_info.flex.csv         # base case with 0.001 flexible loads
- gen_info.flex.high_fossil.csv
- gen_info.flex.no_retire.csv
"""

import sys, os
from pathlib import Path

import pandas as pd

# try to update both the base directory and the _prm version (if present)
for tag in ("", "_prm"):
    # in_dir = Path("switch/in/gas_limit/2030/s4x1_test")
    in_dir = Path(sys.argv[1] + tag)
    if not in_dir.exists():
        continue

    ############
    # Create new versions of gen_info.csv for the scenarios

    gen_info = pd.read_csv(in_dir / "gen_info.csv", na_values=".")

    # TODO: add distillate or diesel, LSFO, petroleum coke, etc., when they appear
    fossil_mask = gen_info["gen_energy_source"].isin(["naturalgas", "coal"])

    # apply a $80/MWh "production tax" to fossil generation to indicate a preference
    # for clean power (roughly equivalent to 0.4 tCO2/MWh (gas) * $200/tCO2 (SCC))
    gen_info_low_fossil = gen_info.copy()
    gen_info_low_fossil.loc[fossil_mask, "gen_variable_om"] += 80

    # prevent age- or economics-based retirement of fossil plants
    gen_info_no_retire = gen_info.copy()
    gen_info_no_retire.loc[fossil_mask, "gen_can_retire_early"] = 0
    gen_info_no_retire.loc[fossil_mask, "gen_max_age"] = 1000

    # apply a $80/MWh "production tax" to new clean generation to indicate
    # a preference for fossil power
    gen_info_high_fossil = gen_info_no_retire.copy()
    # substrings that will identify new-build clean techs but not existing ones
    new_clean_techs = "Battery Storage|LandbasedWind|UtilityPV|OffShoreWind"
    new_clean_mask = gen_info_high_fossil["gen_tech"].str.contains(
        new_clean_techs, case=False, na=False
    )
    gen_info_high_fossil.loc[new_clean_mask, "gen_variable_om"] += 80

    def to_csv(df, file):
        path = in_dir / file
        df.to_csv(path, na_rep=".", index=False)
        print(f"Created {path}.")

    for tag in ["", "high_fossil", "no_retire", "low_fossil"]:
        name = ("gen_info." + tag).strip(".")
        df = locals()[name.replace(".", "_")]
        df_flex = df.copy()
        df_flex.loc[
            df_flex["gen_tech"] == "load_growth", "gen_max_annual_availability"
        ] = 0.001
        if name != "gen_info":
            to_csv(df, name + ".csv")
        to_csv(df_flex, name.replace("gen_info", "gen_info.flex") + ".csv")

    ##########
    # create a scenarios.txt file to simplify running these cases
    if in_dir.parts[0] == "switch":
        # shift reference so it can be run from inside switch dir
        idir = Path(*in_dir.parts[1:])
    else:
        idir = in_dir

    if idir.parts[0] != "in":
        print(
            f'WARNING: Model directory {in_dir} does not start with "switch{os.sep}in{os.sep}" '
            f'or "in{os.sep}"; skipping creation of {in_dir}/scenarios.txt file.'
        )
        continue

    # replace "in" with "out" to get out_dir
    odir = Path("out", *idir.parts[1:])

    main_cases = [
        # least-cost
        f"--scenario-name least_cost --inputs-dir {idir} --outputs-dir {odir}_least_cost "
        f"--input-alias gen_info.csv=gen_info.csv",  # placeholder, will be changed for flex case
        # high fossil build (driven by clean power production tax)
        f"--scenario-name high_fossil_build --inputs-dir {idir} --input-alias gen_info.csv=gen_info.high_fossil.csv "
        f"--outputs-dir {odir}_high_fossil_build ",
        # high fossil (evaluation)
        f"--scenario-name high_fossil --inputs-dir {idir} --input-alias gen_info.csv=gen_info.no_retire.csv "
        f"--include-module study_modules.reuse_build_plan --reuse-dir {odir}_high_fossil_build "
        f"--outputs-dir {odir}_high_fossil",
        # low fossil build (driven by fossil production tax)
        # not as precise as direct minimization, but won't go overboard on
        # offshore wind when it runs out of other options
        f"--scenario-name low_fossil_build --inputs-dir {idir} "
        f"--input-alias gen_info.csv=gen_info.low_fossil.csv "
        f"--outputs-dir {odir}_low_fossil_build",
        # low fossil (evaluation)
        f"--scenario-name low_fossil --inputs-dir {idir} "
        f"--input-alias gen_info.csv=gen_info.csv "
        f"--include-module study_modules.reuse_build_plan --reuse-dir {odir}_low_fossil_build "
        f"--outputs-dir {odir}_low_fossil",
        # min fossil (subject to budget limit); doesn't work well when solar limit is hit
        f"--scenario-name low_fossil_by_budget --inputs-dir {idir} "
        f"--input-alias gen_info.csv=gen_info.csv "
        f"--include-module study_modules.minimize_fossil_power_within_budget --budget-dir {odir}_high_fossil "
        f"--outputs-dir {odir}_low_fossil_by_budget",
    ]
    flex_cases = [
        c.replace(str(odir), str(odir.with_name(odir.name + "_flex")))
        .replace("=gen_info.", "=gen_info.flex.")
        .replace("--scenario-name ", "--scenario-name flex_")
        for c in main_cases
    ]
    scenario_file = in_dir / "scenarios.txt"
    with open(scenario_file, "w") as f:
        f.write("\n".join(main_cases + flex_cases))
    print(f"Created {scenario_file}.")
