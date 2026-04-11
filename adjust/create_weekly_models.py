# %% setup
"""
Create timepoints_nnn.csv and timeseries_nnn.csv defining one-week
models. Also create scenarios_weekly.txt defining scenarios that
use each of these.
"""

import argparse, math, sys, os, shlex
from pathlib import Path
import pandas as pd
import numpy as np

# files to split weeks out of
timepoint_files = {
    # "hydro_timepoints.csv": "timepoint_id",  # not used for this project
    "loads.csv": "TIMEPOINT",
    "variable_capacity_factors.csv": "TIMEPOINT",
    "water_node_tp_flows.csv": "TIMEPOINTS",
    "dr_data.csv": "TIMEPOINT",
    "ee_data.csv": "TIMEPOINT",
}

days_per_group = 7


def parse_arguments():
    # process command line options (name of inputs dir and )
    parser = argparse.ArgumentParser()
    parser.add_argument("in_dir", help="Path to the input directory")
    parser.add_argument(
        "build_dir",
        default="",
        help="Path to the input directory for capacity expansion model whose construction plan should be reused",
    )

    options = parser.parse_args()
    return options


# %% main code
# for testing:
# sys.argv = ['script', 'switch/in/test/2030/p1']


# to profile:
# kernprof -l adjust/create_weekly_models.py switch/in/test/2030/p1
# python -m line_profiler create_weekly_models.py.lprof
# from line_profiler import profile
# @profile


def main():
    options = parse_arguments()
    in_dir = Path(options.in_dir)

    def read(f):
        return pd.read_csv(in_dir / f, na_values=".")

    def write(df, file):
        df.to_csv(in_dir / file, na_rep=".", index=False)
        print(f"saved {in_dir / file}.")

    tp_orig = read("timepoints.csv")
    ts = read("timeseries.csv")

    hours_per_group = days_per_group * 24
    n_groups = math.ceil(len(tp_orig) / hours_per_group)
    n_digits = math.ceil(math.log10(n_groups))
    # next line will fail (by design) if there are multiple timeseries
    tps_per_period = round((ts["ts_num_tps"] * ts["ts_scale_to_period"]).item())

    # assign as many group IDs as needed
    tp_orig["group"] = np.arange(len(tp_orig)) // hours_per_group
    tp_orig = tp_orig.set_index("group")

    print(f"Reading {', '.join(timepoint_files.keys())}")
    files = {}
    for file, col in timepoint_files.items():
        df = read(file)
        # save df and indices for each unique timepoint
        files[file] = (df, df.groupby(col, sort=False).indices)

    scens = []
    for group in range(n_groups):
        tag = f"{group+1:0{n_digits}d}"  # use 1-based tag in file names
        scens.append(tag)

        tp = tp_orig.loc[group, :]
        # stretch each week to match the original study duration
        # (typically 1 year)
        ts["ts_num_tps"] = len(tp)
        ts["ts_scale_to_period"] = tps_per_period / ts["ts_num_tps"]

        write(ts, f"timeseries.{tag}.csv")
        write(tp, f"timepoints.{tag}.csv")

        keys = tp["timepoint_id"].to_numpy(dtype=np.int64, copy=False)

        for file, col in timepoint_files.items():
            # get rows matching current tps, in original order
            # # (have to use .isin for that instead of an index)
            all_rows, indices = files[file]

            # df = all_rows[all_rows[col].isin(tp["timepoint_id"])]
            idx = np.concatenate([indices[k] for k in keys])
            idx.sort()  # use original row order
            df = all_rows.iloc[idx]

            write(df, f"{file[:-4]}.{tag}.csv")

    # create dummy planning_reserve_margin.weekly.csv to use for calculating
    # current PRM level
    lz = read("load_zones.csv")
    prm = lz[["LOAD_ZONE"]].merge(ts[["timeseries"]], how="cross")
    prm["planning_reserve_margin"] = 0.0
    write(prm, "planning_reserve_margin.weekly.csv")

    ##########
    # create scenarios_weekly.txt to simplify running these cases
    if in_dir.parts[0] == "switch":
        # shift reference so it can be run from inside switch dir
        idir = Path(*in_dir.parts[1:])
    else:
        idir = in_dir

    if idir.parts[0] != "in":
        print(
            f'WARNING: Model directory {in_dir} does not start with "switch{os.sep}in{os.sep}" '
            f'or "in{os.sep}"; skipping creation of {in_dir}/scenarios_weekly.txt file.'
        )
        return

    # replace "in" with "out" to get out_dir
    odir = Path("out", *idir.parts[1:])
    scen_file = in_dir / "scenarios_weekly.txt"

    alias_files = ["timeseries.csv", "timepoints.csv"] + list(timepoint_files.keys())

    # define sets of scenarios for high_fossil, high_renewable and high_renewable_flex cases
    meta_scens = parse_build_args(options)

    with open(scen_file, "w") as f:
        for scen, args in meta_scens.items():
            for tag in scens:
                aliases = " ".join(
                    f"{file}={file[:-4]}.{tag}.csv" for file in alias_files
                )
                f.write(
                    f"--scenario-name {scen}_{tag} --inputs-dir {idir} --outputs-dir {odir / scen / tag} "
                    # base arguments for the meta scenario (e.g., high_renwewable_flex)
                    f"{args} "
                    # arguments for this weekly sub-model
                    f"--input-aliases {aliases} "
                    # reduce output
                    "--include-module study_modules.reduce_reporting "
                    "--skip-generic-output "
                    "--skip-output-files dispatch.csv dispatch_wide.csv dispatch_gen_annual_summary.csv dispatch_annual_summary_fuel.pdf dispatch_annual_summary_tech.pdf "
                    # relax environmental policies
                    "--exclude-modules study_modules.min_capacity_constraint study_modules.max_capacity_constraint study_modules.rps_regional "
                    # don't worry about scheduled outages; we assume they will not occur during critical periods
                    "--exclude-module study_modules.scheduled_outages "
                    # allow unserved load, to avoid infeasibility (replaced by
                    # calculating a potentially negative current reserve margin)
                    # "--include-module study_modules.unserved_load "
                    # "--exclude-module study_modules.planning_reserves_extreme_days "
                    # find current PRM, possibly negative (will be used to
                    # adjust PRM in capacity expansion stage if needed)
                    "--input-alias planning_reserve_margin.csv=planning_reserve_margin.weekly.csv "
                    "--find-current-prm "
                    "\n"
                )
    print(f"created {scen_file}")

    # note: each weekly set needs to be run a little differently, e.g.,
    """
    switch solve-scenarios --scenario-list in/2030/weekly/scenarios_weekly.txt --    
    --input-alias gen_info.csv=gen_info.flex.csv --include-module study_modules.demand_response_investment --include-module study_modules.efficiency_investment
    """


def parse_build_args(options):
    """
    Parses a string of command-line style arguments to extract specific options
    (--scenario-name, --inputs-dir, --outputs-dir, --reuse-dir) into a namespace,
    and returns the namespace along with a string of all unused arguments.

    Parameters:
    - scenario_str: A string containing space-separated arguments.

    Returns:
    - A tuple: (namespace, unused_args_str)
      - namespace: An argparse.Namespace object with attributes scenario_name, inputs_dir, outputs_dir, reuse_dir (None if not provided).
      - unused_args_str: A string of the unused arguments, joined by spaces.
    """
    scen_names = ["high_fossil", "high_renewable", "high_renewable_flex"]

    default_settings = {"base": ""}
    if not options.build_dir:
        return default_settings

    with open(Path(options.build_dir) / "scenarios.txt") as f:
        scenario_strings = f.read().splitlines()

    # Create an ArgumentParser for the arguments we want to use here and/or drop from the scenario definition
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-name")
    parser.add_argument("--inputs-dir", default=None)
    parser.add_argument("--outputs-dir")
    parser.add_argument("--reuse-dir", default=None)

    settings = {}
    for scen_str in scenario_strings:
        # drop "--include-module study_modules.reuse_build_plan", but only if it
        # comes in exactly that form (parsing --include-module more generally is
        # complex and not really needed here)
        scen_str = scen_str.replace(
            "--include-module study_modules.reuse_build_plan", ""
        ).replace("  ", " ")
        # change references to gen_info.flex.*.csv to gen_info.flex_weekly.*.csv
        # (allow concentrated use of interruptible load)
        scen_str = scen_str.replace("gen_info.flex.", "gen_info.flex_weekly.")

        # parse the arguments we want to use/drop and retain the rest
        info, old_args = parser.parse_known_args(shlex.split(scen_str))

        if info.scenario_name not in scen_names:
            continue

        new_args = [
            "--include-module",
            "study_modules.reuse_build_plan",
            "--reuse-dir",
            info.outputs_dir,
        ]
        settings[info.scenario_name] = shlex.join(new_args + old_args)

    if settings:
        return settings
    else:
        return default_settings


if __name__ == "__main__" and "ipykernel" not in sys.argv[0]:
    # running as a script, not from a jupyter environment
    main()

# %%
