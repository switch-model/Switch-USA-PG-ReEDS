"""
Create alternative versions of gen_info.csv that have various permutations of

- a clean power production tax (to drive more fossil investment)
- no retirement of existing generators (to sustain fossil infrastructure)
- flexibility for grown loads

These will be:

- gen_info.high_fossil.csv  # used for high_fossil.build case
- gen_info.no_retire.csv    # used for high_fossil (cost evaluation) case
# - gen_info.flex.csv         # base case with 0.001 flexible loads
# - gen_info.flex.high_fossil.csv
# - gen_info.flex.no_retire.csv
"""

import sys, os, re
from pathlib import Path

import pandas as pd


def to_csv(df, file):
    path = in_dir / file
    df.to_csv(path, na_rep=".", index=False)
    print(f"Created {path}.")


# try to update both the base directory and the _prm version (if present)
for tag in ("", "_prm"):
    # in_dir = Path("switch/in/gas_limit/2030/s4x1_test")
    in_dir = Path(sys.argv[1] + tag)
    if not in_dir.exists():
        continue

    ############
    # Create new versions of gen_info.csv for the scenarios

    gen_info = pd.read_csv(in_dir / "gen_info.csv", na_values=".")

    # TODO: add LSFO if it ever appears
    # note: we don't apply fossil_mask to petroleum coke or petroleum liquids
    # (distillate) plants because we assume they have other reasons for running
    # or retiring
    fossil_mask = gen_info["gen_energy_source"].isin(["naturalgas", "coal"])

    # # apply a $80/MWh "production tax" to fossil generation to indicate a preference
    # # for clean power (roughly equivalent to 0.4 tCO2/MWh (gas) * $200/tCO2 (SCC))
    # gen_info_low_fossil = gen_info.copy()
    # gen_info_low_fossil.loc[fossil_mask, "gen_variable_om"] += 80

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

    for tag in ["", "high_fossil", "no_retire"]:  # , "low_fossil"]:
        name = ("gen_info." + tag).strip(".")
        df = locals()[name.replace(".", "_")]
        if name != "gen_info":
            to_csv(df, name + ".csv")
        # 0.1% interruptible load (no longer used because difficult to decide
        # how it affects PRM or integrate into resource adequacy evaluation)
        # df_flex = df.copy()
        # df_flex.loc[
        #     df_flex["gen_tech"] == "load_growth", "gen_max_annual_availability"
        # ] = 0.001
        # to_csv(df_flex, name.replace("gen_info", "gen_info.flex") + ".csv")
        # # allow full flexibility to be concentrated in each weekly case
        # # if needed (for rare emergencies)
        # df_flex.loc[
        #     df_flex["gen_tech"] == "load_growth", "gen_max_annual_availability"
        # ] *= 52
        # to_csv(df_flex, name.replace("gen_info", "gen_info.flex_weekly") + ".csv")

    # if this is a low_growth case and there already exists a regular case with
    # the same name, create loads.low_growth.csv in the regular case, using the
    # same time sampling as the regular case, but with peak and average matched
    # to this case
    reg_dir = in_dir.with_name(in_dir.name.replace("_low_growth", ""))
    if in_dir != reg_dir and reg_dir.exists():
        load = {}
        load_stats = {}
        for d in (in_dir, reg_dir):
            tp = (
                pd.read_csv(d / "timepoints.csv", na_values=".")
                .merge(
                    pd.read_csv(d / "timeseries.csv", na_values="."), on="timeseries"
                )
                .merge(
                    pd.read_csv(d / "periods.csv", na_values="."),
                    left_on="ts_period",
                    right_on="INVESTMENT_PERIOD",
                )
                .eval(
                    "tp_weight=ts_duration_of_tp * ts_scale_to_period / (period_start - period_end + 1)"
                )
                .set_index("timepoint_id")
            )
            l = pd.read_csv(d / "loads.csv", na_values=".")
            l["tp_weight"] = l["TIMEPOINT"].map(tp["tp_weight"])
            l["PERIOD"] = l["TIMEPOINT"].map(tp["INVESTMENT_PERIOD"])
            s = (
                l.groupby(["LOAD_ZONE", "PERIOD"])[["zone_demand_mw"]]
                .max()
                .rename(columns={"zone_demand_mw": "peak_mw"})
            )
            s["avg_mw"] = (l["zone_demand_mw"] * l["tp_weight"]).groupby(
                [l["LOAD_ZONE"], l["PERIOD"]]
            ).sum() / l.groupby(["LOAD_ZONE", "PERIOD"])["tp_weight"].sum()
            load[d] = l
            load_stats[d] = s

        scale = (load_stats[in_dir]["peak_mw"] - load_stats[in_dir]["avg_mw"]) / (
            load_stats[reg_dir]["peak_mw"] - load_stats[reg_dir]["avg_mw"]
        )
        offset = load_stats[in_dir]["avg_mw"] - scale * load_stats[reg_dir]["avg_mw"]
        adj = pd.DataFrame({"scale": scale, "offset": offset})
        loads_new = load[reg_dir].join(adj, on=["LOAD_ZONE", "PERIOD"])
        loads_new["zone_demand_mw"] = loads_new.eval("zone_demand_mw * scale + offset")
        loads_new[["LOAD_ZONE", "TIMEPOINT", "zone_demand_mw"]].to_csv(
            reg_dir / "loads.low_growth.csv", index=False, na_rep="."
        )
        print(f"Created {reg_dir / 'loads.low_growth.csv'}   <--- in base-case dir")
        # can verify these match with
        # q "select sum(avg_mw), sum(max_mw) from (select avg(zone_demand_mw) as avg_mw, max(zone_demand_mw) as max_mw from 'loads.csv' group by load_zone);"

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

    # parameters used for capacity expansion model, choosing what to build
    build_cases = {
        "high_fossil_build": [  # make build plan
            "--input-alias gen_info.csv=gen_info.high_fossil.csv",
        ],
        "high_renewable_build": [
            "--include-module study_modules.solar_push --total-solar-gw 500",
        ],
        "high_renewable_flex_build": [
            "--include-module study_modules.solar_push --total-solar-gw 500",
            # "--input-alias gen_info.csv=gen_info.flex.csv",
            "--include-module study_modules.demand_response_investment",
            "--include-module study_modules.efficiency_investment",
        ],
    }

    # parameters used when re-running capacity expansion model with frozen
    # construction decisions (including possibly resource-adequacy adjustments)
    # to evaluate final costs
    # for this study, these are the same as the build cases, except we turn off
    # the anti-clean-power bias in the fossil case
    eval_cases = {
        "high_fossil": [
            s.replace("gen_info.high_fossil.csv", "gen_info.no_retire.csv")
            for s in build_cases["high_fossil_build"]
        ],
        "high_renewable": build_cases["high_renewable_build"].copy(),
        "high_renewable_flex": build_cases["high_renewable_flex_build"].copy(),
    }
    # reuse plan from correct output directory (e.g., out/2030/s20x1/high_renewable)
    for c, args in eval_cases.items():
        args.append(
            f"--include-module study_modules.reuse_build_plan --reuse-dir {odir / c}_build"
        )

    peak_fuel_cases = {
        f"peak_fuel_{c}": args
        + [f"--input-alias fuel_cost.csv=../{idir.name}_peak_fuel/fuel_cost.csv"]
        for c, args in eval_cases.items()
    }

    low_growth_cases = {
        f"low_growth_{c}": args + [f"--input-alias loads.csv=loads.low_growth.csv"]
        for c, args in eval_cases.items()
    }

    scenario_groups = {
        "build": [build_cases],
        "eval": [eval_cases, peak_fuel_cases, low_growth_cases],
    }

    # make case defs: {'suffix': ['case 1 def', 'case 2 def', ...]}
    cases = {}
    for suffix, case_dicts in scenario_groups.items():
        cases[suffix] = []
        for c in case_dicts:
            for scen, args in c.items():
                a = [
                    f"--scenario-name {scen} --inputs-dir {idir} --outputs-dir {odir / scen}"
                ] + args
                cases[suffix].append(" ".join(a))

    # create scenarios_build.txt and scenarios_eval.txt
    for suffix, args in cases.items():
        scenario_file = in_dir / f"scenarios_{suffix}.txt"
        with open(scenario_file, "w") as f:
            f.writelines(line + "\n" for line in args)
        print(f"Created {scenario_file}.")
