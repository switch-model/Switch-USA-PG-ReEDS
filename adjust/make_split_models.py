# %% setup
"""
Split the specified model directory into separate models for every timeseries,
for use in production cost modeling or reliability analysis.

Each model is defined by new files like timepoints.nnn.csv and loads.nnn.csv
created in the original directory. This script also creates scenarios_split.txt
to setup and run each of these separate models.

The script can optionally use the output from capacity expansion models defined
in a different scenarios.txt file as construction plans for the split models
(see help for `--reuse-build-scenarios` argument).

This script also adds any extra arguments passed on the command line to the
scenario definitions for the split model. This can be useful for reducing model
output or adjusting reliability mechanisms.
"""

import argparse, math, sys, os, shlex, shutil
from pathlib import Path
import pandas as pd
import numpy as np

# files to split weeks out of (also see add_extreme_days.py)
timepoint_files = {
    "loads.csv": "TIMEPOINT",
    "variable_capacity_factors.csv": "TIMEPOINT",
    "water_node_tp_flows.csv": "TIMEPOINTS",
    "dr_data.csv": "TIMEPOINT",
    "ee_data.csv": "TIMEPOINT",
}
# large files not needed for this study
exclude_files = {
    "graph_timestamp_map.csv",
    "hydro_timepoints.csv",
    "hydro_timeseries.csv",
}

split_scenario_file = f"scenarios_split.txt"


def parse_script_args():
    # process command line options for this script itself ()
    parser = argparse.ArgumentParser()
    parser.add_argument("in_dir", help="Path to the input directory")
    parser.add_argument(
        "--reuse-build-scenarios",
        default="",
        help="""
            Path to scenarios.txt file with scenarios that should be used as the
            basis for the split-model scenarios. These are typically the
            scenarios used for the capacity expansion stage. If provided, each
            scenario in the original scenarios.txt will be repeated for each
            separate timeseries in scenarios_split.txt, and each of these will
            have arguments added to reuse the output from the original scenario
            as the construction plan for the split scenario, via
            --reuse-build-plan. Will be omitted from scenario definition if not
            specified.
        """,
    )
    return parser.parse_known_args()


# %% main code
# for testing:
# sys.argv = ['script', 'switch/in/test/2030/p1']
# sys.argv = ['script'] + shlex.split("switch/in/2030/resource_adequacy --reuse-build-scenarios switch/in/2030/s40x1/scenarios_build.txt --include-module study_modules.reduce_reporting --skip-generic-output --skip-output-files dispatch.csv dispatch_wide.csv dispatch_gen_annual_summary.csv dispatch_annual_summary_fuel.pdf dispatch_annual_summary_tech.pdf --exclude-modules study_modules.min_capacity_constraint study_modules.max_capacity_constraint study_modules.rps_regional --exclude-module study_modules.scheduled_outages --exclude-module study_modules.planning_reserves_extreme_days --include-module study_modules.unserved_load")


def clone_to_split_dir(in_dir, tag):
    # create in_dir/tag folder with hard links to in_dir files
    base_dir = Path(in_dir)
    split_dir = base_dir / tag
    if split_dir.is_dir():
        shutil.rmtree(split_dir)
    split_dir.mkdir(parents=True)
    # link all root-level non-time files into the split dir
    for p in base_dir.iterdir():
        if (
            p.is_file()
            and p.name not in timepoint_files
            and p.name not in {"timeseries.csv", "timepoints.csv"}
            and p.name not in exclude_files
        ):
            # make hard link: faster and smaller than copying
            # shutil.copy2(p, split_dir / p.name)
            (split_dir / p.name).hardlink_to(p)
    # print(f"Created {split_dir} model directory")


def main():
    script_options, shared_args = parse_script_args()
    in_dir = Path(script_options.in_dir)

    def read(f):
        return pd.read_csv(in_dir / f, na_values=".")

    def write_split(tag, file, df):
        df.to_csv(in_dir / tag / file, na_rep=".", index=False)
        # print(f"saved {in_dir / tag / file}.")

    tp = read("timepoints.csv").set_index("timeseries", drop=False)
    ts = read("timeseries.csv").set_index("timeseries", drop=False)

    period_hours = (
        (ts["ts_num_tps"] * ts["ts_duration_of_tp"] * ts["ts_scale_to_period"])
        .groupby(ts["ts_period"])
        .sum()
    ).to_dict()

    # map group numbers to timeseries
    group_ts = dict(enumerate(ts["timeseries"]))

    n_groups = len(group_ts)
    n_digits = math.ceil(math.log10(n_groups))

    print(f"Reading {', '.join(timepoint_files.keys())}")
    files = {}
    for file, col in timepoint_files.items():
        df = read(file)
        # save df and indices for each unique timepoint
        files[file] = (df, df.groupby(col, sort=False).indices)

    print(f"Creating split model subdirectories ({in_dir}/{'N'*n_digits}):")
    scens = []
    for group in range(n_groups):
        tag = f"{group:0{n_digits}d}"  # use 1-based tag in file names
        scens.append(tag)

        # make sure the directory for the split model exists
        clone_to_split_dir(in_dir, tag)

        cur_ts = group_ts[group]

        tp_split = tp.loc[[cur_ts], :]
        ts_split = ts.loc[[cur_ts], :]
        # stretch each split timeseries to match the original study duration
        # (typically 1 year)
        ts_hours = (ts_split["ts_num_tps"] * ts_split["ts_duration_of_tp"]).sum()
        ts_split["ts_scale_to_period"] = (
            period_hours[ts_split["ts_period"].iloc[0]] / ts_hours
        )

        write_split(tag, "timeseries.csv", ts_split)
        write_split(tag, "timepoints.csv", tp_split)

        keys = tp_split["timepoint_id"].to_numpy(dtype=np.int64, copy=False)

        for file, col in timepoint_files.items():
            # get rows matching current tps, in original order
            all_rows, indices = files[file]
            idx = np.concatenate([indices[k] for k in keys])
            idx.sort()  # use original row order
            df = all_rows.iloc[idx]
            write_split(tag, file, df)

        if group == 0 or (group + 1) % (n_groups / 10) < 1:
            print(
                f"Created {group+1}/{n_groups} model subirectories ({(group+1)/n_groups:.0%})"
            )

    ##########
    # create scenario file to simplify running these cases
    if in_dir.parts[0] == "switch":
        # shift reference so it can be run from inside switch dir
        idir = Path(*in_dir.parts[1:])
    else:
        idir = in_dir

    if idir.parts[0] != "in":
        print(
            f'WARNING: Model directory {in_dir} does not start with "switch{os.sep}in{os.sep}" '
            f'or "in{os.sep}"; skipping creation of {in_dir}/{split_scenario_file} file.'
        )
        return

    # replace "in" with "out" to get out_dir
    odir = Path("out", *idir.parts[1:])
    scen_file = in_dir / split_scenario_file

    # define sets of scenarios for all construction plans
    build_scens = make_build_scens(script_options.reuse_build_scenarios, shared_args)

    with open(scen_file, "w") as f:
        for scen, args in build_scens.items():
            for tag in scens:
                f.write(
                    # scenario name and data for this split
                    f"--scenario-name {scen}_{tag} --inputs-dir {idir / tag} --outputs-dir {odir / scen / tag} "
                    # arguments for this capacity expansion scenario (e.g., high_renwewable_flex)
                    f"{args} "
                    "\n"
                )
    print(f"created {scen_file}")


def make_build_scens(reuse_build_scenarios, shared_args):
    """
    Creates a dict with template arguments for each build scenario (capacity
    expansion model), reusing arguments used for the build scenario (e.g.,
    generator parameters) and usually adding arguments to reuse the construction
    plan created by that model. This will also append any extra arguments
    specified on the command line for this script.

    Parameters:
    - reuse_build_scenarios: path to scenarios.txt file that holds definitions
      for build scenarios (capacity expansion models whose construction plan
      will be used for the split models). If "", no construction plan will be
      reused and user should add arguments to the call to this script or at
      runtime to controle the construction plan.
    - shared_args: a list of other command-line arguments that were passed to
      this script, which will be applied to all the split scenarios

    Returns:
    - build_scens: a dict of "base scenario name": "scenario string" representing
      arguments to use as the starting point for each split scenario (everything
      except --scenario-name and --inputs-dir)
    """
    # Create an ArgumentParser for the arguments we want to use here and/or drop
    # from the scenario definition (these will be redefined later)
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-name")
    parser.add_argument("--inputs-dir", default=None)
    parser.add_argument("--outputs-dir", default="outputs")

    if reuse_build_scenarios:
        with open(reuse_build_scenarios) as f:
            scenario_strings = f.read().splitlines()
    else:
        # User didn't specify build scenarios to use; act as if there is a
        # minimal scenario file that just defines one "base" scenario, and skip
        # the reuse-build-plan parts.
        scenario_strings = ["--scenario-name base"]

    build_scens = {}
    for scen_str in scenario_strings:
        # parse the arguments we want to use and retain the rest for the scenario definition
        local_options, base_scen_args = parser.parse_known_args(shlex.split(scen_str))
        if reuse_build_scenarios:
            reuse_build_plan_args = [
                "--include-module",
                "study_modules.reuse_build_plan",
                "--reuse-dir",
                local_options.outputs_dir,
            ]
        else:
            reuse_build_plan_args = []

        # define a scenario that uses the --reuse-dir args, most args from the
        # base scenario definition, and any shared arguments for all scenarios
        # that were specified on the command line
        build_scens[local_options.scenario_name] = shlex.join(
            reuse_build_plan_args + base_scen_args + shared_args
        )

    return build_scens


if __name__ == "__main__" and "ipykernel" not in sys.argv[0]:
    # running as a script, not from a jupyter environment
    main()

# %%
