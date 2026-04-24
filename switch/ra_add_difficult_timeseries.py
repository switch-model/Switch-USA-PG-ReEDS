"""
This script checks whether resource adequacy (RA) models found any unserved
load, and if so, adds the timeseries with the most unserved load into the
capacity expansion (CE) models. It also labels CE models as "adequate",
"inadquate", or "infeasible".

This script will create RA versions of the CE models, i.e., new copies in
ce_dir/ra/ce_scenario_name that have additional, difficult timeseries added with
0 weight and the requested planning reserve margin.

Please note: this script overwrites the scenario list files if they end with _ra
before the extension and the existing CE directory if its parent directory is
called ra. So the files and folders used for the base model first iteration
should not have this pattern.

This script is designed to be called iteratively. For the example below, the
main CE models are defined in `in/build` and saved in
`out/build/{scen1,scen2,...}`, the split/RA models are in
`in/resource_adequacy`, the definitions for the CE scenarios are in
`in/build/scenarios_build.txt` and the definitions for the RA split models are
in `in/resource_adequacy/scenarios_split.txt`.

Then this script should be used as follows:

    delete in/build/ra if present
    loop:
        if first iteration:
            ce_scens = in/build/scenarios_build.txt
            ra_scens = in/resource_adequacy/scenarios_split.txt
        else:
            # _ra version of scenario lists are created by this script
            # and show only the remaining CE cases that still have
            # unserved load and haven't been identified as infeasible
            ce_scens = in/build/scenarios_build_ra.txt
            ra_scens = in/resource_adequacy/scenarios_split_ra.txt

        switch solve-scenarios --scenario-list $ce_scens
        switch solve-scenarios --scenario-list $ra_scens

        if first iteration and want backup of initial plan:
            copy `out/build/scen*` to backup location

        python ra_add_difficult_timeseries.py --ce-scens-file $ce_scens --ra-scens-file $ra_scens

        if `in/build/scenarios_build_ra.txt` is empty:
            break

Additional notes:

For the example above, after the script runs, RA versions of the CE models,
including the extra, difficult timeseries, will be in
`in/build/ra/{scen1,scen2,...}` and the scenarios that need to be run for the
next step of the iteration will be defined in `in/build/scenarios_build_ra.txt`

After `--max-prm-timeseries-count` is reached, PRMs will be multiplied by 1.25
(rising by at least 1%) in each iteration until there is no unserved load or
until `--max-prm-level` is reached. If `--max-prm-level` is reached, the
scenario will be marked as "infeasible" in `out/build/scen?/ra_status.txt` and
iteration will stop.

If there is unserved load, but only in timeseries that have already been added
to the RA version of the CE model, then the iteration will switch over to
raising the PRM, as if --max-prm-timeseries-count had been reached. Then it will
revert to adding difficult timeseries in the next iteration if there is still
unserved load. This generally should not occur.
"""

import argparse, shlex, shutil, sys, os
from pathlib import Path
import pandas as pd


# patch pd.DataFrame.to_csv() to reduce binary-csv artifacts,
# e.g., 0.0387 -> 0.038700000000000005
# (same as pg_to_switch.py)
def clean_to_csv(self, *args, float_format="%.15g", **kwargs):
    pd_to_csv(self, *args, float_format=float_format, **kwargs)


try:
    pd_to_csv  # type: ignore
except:
    pd_to_csv = pd.DataFrame.to_csv
    pd.DataFrame.to_csv = clean_to_csv


# same list as in adjust/make_split_models.py or adjust/add_extreme_days.py
timepoint_files = {
    # "hydro_timepoints.csv": "timepoint_id",  # not used for this project
    "loads.csv": "TIMEPOINT",
    "variable_capacity_factors.csv": "TIMEPOINT",
    "water_node_tp_flows.csv": "TIMEPOINTS",
    "dr_data.csv": "TIMEPOINT",
    "ee_data.csv": "TIMEPOINT",
}

# possible status markers (use vars instead of strings to avoid false mismatches)
adequate = "adequate"
inadequate = "inadequate"
stalled = "stalled"
infeasible = "infeasible"


def get_scenario_args(scenario_file):
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-name")
    parser.add_argument("--inputs-dir")
    parser.add_argument("--outputs-dir")

    with open(scenario_file) as f:
        scenario_strings = f.read().splitlines()

    scen_info = {}
    for scen_str in scenario_strings:
        if scen_str.startswith("#"):
            continue
        options, other_args = parser.parse_known_args(shlex.split(scen_str))
        d = scen_info[options.scenario_name] = vars(options)
        d["args"] = other_args

    return scen_info


import argparse


def get_script_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ce-scens-file",
        type=str,
        help="""
            Scenario list file used to define capacity expansion scenarios,
            e.g., in/2030/s20x1/scenarios_build.txt (usually created by
            adjust/define_scenarios.py)
        """,
    )
    parser.add_argument(
        "--ra-scens-file",
        type=str,
        help="""
            Scenario list file used to define resource adequacy (split)
            scenarios, e.g., in/2030/resource_adequacy/scenarios_split.txt
            (usually created by adjust/create_split_models.py)
        """,
    )
    parser.add_argument(
        "--timepoint-duration",
        default=2,
        type=int,
        help="""
            Number of hours to use between timepoints when adding difficult
            timeseries from the resource adequacy models to the capacity
            expansion models (default is 2)
        """,
    )
    parser.add_argument(
        "--initial-prm",
        default=0.02,
        type=float,
        help="""
            Planning reserve margin (decimal fraction) to apply to loads on
            difficult timeseries that are added to the capacity expansion models
            (default is 0.02, i.e, 2%%). This is the starting point, and the PRM
            level will be increased if the model is not adequate after the
            maximum number of difficult timeseries are added.
        """,
    )
    parser.add_argument(
        "--max-prm-timeseries-count",
        default=20,
        type=int,
        help="""
            Maximum number of difficult timeseries that can be included in the
            capacity expansion model before the iteration switches from adding
            difficult days to increasing the planning reserve margin.
            (including any previously added to the
            planning reserve margin system by adjust/add_extreme_days.py)
        """,
    )
    parser.add_argument(
        "--max-prm-level",
        default=1.0,
        type=float,
        help="""
            Highest level that the planning reserve margin can be reach before
            giving up and marking the scenario as infeasible. This should be a
            decimal fraction (e.g., 1.0, indicating PRM can't be raised 100%%,
            i.e., the system must be ready to serve double the normal load on
            that day).
        """,
    )
    options = parser.parse_args()
    return options


# testing:
"""
options = lambda: None # dummy namespace object
options.ce_scens_file = "in/2030/s20x1/scenarios_build.txt"
options.ra_scens_file = "in/2030/resource_adequacy/scenarios_split.txt"
options.timepoint_duration = 2
options.initial_prm = 0.02
options.max_prm_timeseries_count = 20
"""

# note: this script deals with three kinds of models:
# - CE model: original capacity expansion model that will be left untouched
# - CE RA model: the resource adequacy version of the original CE model; this
#   starts as a copy of the original CE model, but then has difficult timeseries
#   incrementally added to it
# - RA model: models where individual timeseries (or possibly longer durations)
#   are split into separate models, and construction decisions are frozen to
#   match the output from a CE model or CE RA model. These are run to check
#   for unserved load in each timeseries.


def main(options):
    ce_scens = get_scenario_args(options.ce_scens_file)
    ra_scens = get_scenario_args(options.ra_scens_file)

    # We could just report a message, but throwing an error is more likely to stop
    # an iterative loop, which should never reach here.
    if not ce_scens:
        raise RuntimeError(f"No scenarios are defined in {options.ce_scens_file}.")
    if not ra_scens:
        raise RuntimeError(f"No scenarios are defined in {options.ra_scens_file}.")

    def ce_ra_path(ce_scen):
        in_dir = Path(ce_scens[ce_scen]["inputs_dir"])
        if in_dir.parent.name == "ra":
            # model was run from a CE RA directory
            return in_dir
        else:
            # model was run from the base CE directory,
            # but the next model should run from the
            # subdirectory with the RA version of this
            # CE model
            return in_dir / "ra" / ce_scen

    def make_ce_ra_dir(ce_scen):
        # create ce_dir/ra/ce_scen_name folder if it doesn't exist already
        ce_dir = Path(ce_scens[ce_scen]["inputs_dir"])
        ce_ra_dir = ce_ra_path(ce_scen)  # possibly the same dir, on re-runs
        if ce_dir == ce_ra_dir:
            # reusing an existing CE RA dir; double-check that it really is a CE
            # RA dir and not a misnamed CE dir (something/ra/model, but not
            # meant to be the RA version of a CE model)
            if not (ce_dir.parent.parent / "switch_inputs_version.txt").is_file():
                raise RuntimeError(
                    f"This script will treat {ce_ra_dir} as a CE RA model directory "
                    "because its parent's name is 'ra'. However its grandparent "
                    f"{ce_ra_dir.parent.parent} is not a model directory, so this "
                    "appears to be a normal model, not the RA version of a CE model. "
                    "Terminating to avoid overwriting a normal model directory. "
                    "To avoid this error, please rename the parent to something "
                    "other than 'ra'."
                )
        else:
            # first run; create the RA versions of the CE dir
            if ce_ra_dir.is_dir():
                # ra subdir exists, probably from a previous run
                print(f"deleting pre-existing {ce_ra_dir}")
                shutil.rmtree(ce_ra_dir)
            # copy all files (but not subdirs) into the ce ra dir
            ce_ra_dir.mkdir(parents=True)
            for p in ce_dir.iterdir():
                if p.is_file():
                    shutil.copy2(p, ce_ra_dir / p.name)
            print(f"Created CE RA model directory `{ce_ra_dir}`.")

        # first run, copy to the new dir
        # if ce_ra_dir.is_dir():
        #     # ra subdir exists (should be rare); warn if this is older than
        #     # parent, which generally indicates they recreated the main model
        #     # dir without deleting the ra subdir
        #     older = []
        #     missing = []
        #     for p in ce_dir.iterdir():
        #         if p.is_file():
        #             q = ce_ra_dir / p.name
        #             if q.is_file():
        #                 if p.stat().st_mtime > q.stat().st_mtime:
        #                     older.append(p.name)
        #             else:
        #                 missing.append(p.name)
        #     if older:
        #         print(
        #             f"WARNING: the following files already exist in `{ce_ra_dir}` "
        #             f"but are older than the versions in `{ce_dir}`: "
        #             f"{', '.join(older)}. "
        #             f"The `{ce_ra_dir}` directory should be deleted whenever `{ce_dir}` is rebuilt."
        #         )
        #     if missing:
        #         print(
        #             f"WARNING: the following files exist in `{ce_dir}` but not in "
        #             f"`{ce_ra_dir}`: {', '.join(missing)}. "
        #         )
        # else:
        #     # dir doesn't exist; create it and copy all files (but not subdirs)
        #     # into the ra dir
        #     ce_ra_dir.mkdir(parents=True, exist_ok=True)
        #     for p in ce_dir.iterdir():
        #         if p.is_file():
        #             shutil.copy2(p, ce_ra_dir / p.name)
        #     print(f"Created `{ce_ra_dir}` model directory")

    def ra_split_tag(ra_scen):
        # parse ra split number (e.g., 0123) from an ra scenario name
        # (currently this is everything after the final underscore)
        # If this changes, also change f"{ce_scen}_{tag}" code below
        return ra_scen.rsplit("_", 1)[1]

    def read_ce_ra_in(ce_scen, f):
        # read a file from the ra version of the ce input directory
        # we turn off low_memory when reading to handle mixed column types in
        # timeseries.csv
        file = ce_ra_path(ce_scen) / f
        return pd.read_csv(file, na_values=".", low_memory=False)

    def write_ce_ra_in(ce_scen, f, df):
        # write a file in the ra version of the ce input directory
        file = ce_ra_path(ce_scen) / f
        df.to_csv(file, na_rep=".", index=False)
        print(f"saved {file}.")

    def read_ra_in(ra_scen, f):
        # read an input file for an ra model from ra_main_dir / NNNN
        # note: if the lookup logic changes, also update tp_ts code below
        file = Path(ra_scens[ra_scen]["inputs_dir"]) / f
        return pd.read_csv(file, na_values=".")

    def read_ra_out(ra_scen, f):
        # read a ra-scen output
        file = Path(ra_scens[ra_scen]["outputs_dir"]) / f
        return pd.read_csv(file, na_values=".")

    # cross-reference ce scenarios and ra scenarios
    for ra_scen, info in ra_scens.items():
        ce_scen = ra_scen.rsplit("_", 1)[0]
        info["ce_scen"] = ce_scen
        ce_scens[ce_scen].setdefault("ra_scens", []).append(ra_scen)

    # # testing
    # print("=========================================")
    # print("only using high_fossil_build for now!!!!!")
    # print("=========================================")
    # ce_scens = {k: v for k, v in ce_scens.items() if k == "high_fossil_build"}
    # ra_scens = {
    #     k: v for k, v in ra_scens.items() if v["ce_scen"] == "high_fossil_build"
    # }

    # make sure the RA versions of the CE dir exist
    for ce_scen in ce_scens.keys():
        make_ce_ra_dir(ce_scen)

    # get mappings for timepoint -> timeseries and timeseries -> tag ('0123')
    # (could relax this assertion in later versions, but it's efficient to just
    # read the ra files once if we can get away with it)
    ra_in_dirs = {
        ce_scen: {ra_scens[ra_scen]["inputs_dir"] for ra_scen in bs_info["ra_scens"]}
        for ce_scen, bs_info in ce_scens.items()
    }
    v0 = next(iter(ra_in_dirs.values()))
    assert all(
        v == v0 for v in ra_in_dirs.values()
    ), "need to rewrite for multiple ra inputs dirs"
    # choose a single reference ce scen and matching ra scens to use for looking
    # up timepoint, timeseries and tag data
    ref_ce_scen = next(iter(ce_scens.keys()))
    ref_ra_scens = ce_scens[ref_ce_scen]["ra_scens"]
    tp_ts = {}
    ts_tag = {}
    for ra_scen in ref_ra_scens:
        # same logic as read_ra_in()
        tp = read_ra_in(ra_scen, "timepoints.csv")
        tp_ts.update(tp.set_index("timepoint_id")["timeseries"])
        tag = ra_split_tag(ra_scen)
        for ts in tp["timeseries"].unique():
            ts_tag[ts] = tag

    # find worst day that isn't already included in each ce scenario
    # test: ce_scen = 'high_fossil_build'; ce_info = ce_scens[ce_scen]
    for ce_scen, ce_info in ce_scens.items():
        ce_df = read_ce_ra_in(ce_scen, "timeseries.csv")
        # get list of ra scens in this ce scen
        rs = ce_scens[ce_scen]["ra_scens"]

        # get average unserved load for each timeseries
        # read in all unserved load files, getting total for each timepoint across all zones
        unserved_load_dfs = []
        n_scens = len(rs)
        print(f"Reading unserved load files for {ce_scen}:")
        for i, r in enumerate(rs):
            try:
                # df = (
                #     pd.read_csv(
                #         Path(ra_scens[r]["outputs_dir"]) / "unserved_load.csv",
                #         usecols=["TIMEPOINT", "UnservedLoadMW"],
                #         dtype={"TIMEPOINT": "int32", "UnservedLoadMW": "float32"},
                #         # na_filter=False,
                #         # engine="c",  # or "c" if pyarrow is unavailable / incompatible
                #         engine="pyarrow",  # or "c" if pyarrow is unavailable / incompatible
                #     )
                #     .groupby("TIMEPOINT", sort=False)["UnservedLoadMW"]
                #     .sum()
                #     .reset_index()
                # )
                df = (
                    read_ra_out(r, "unserved_load.csv")
                    .groupby("TIMEPOINT")["UnservedLoadMW"]
                    .sum()
                    .reset_index()
                )
                unserved_load_dfs.append(df)
            except (FileNotFoundError, TimeoutError):
                print(f"  Unable to read unserved load file for case {r}.")
            if i == 0 or (i + 1) % (n_scens / 10) < 1:
                print(f"  Read {i+1}/{n_scens} files ({(i+1)/n_scens:.0%}).")

        unserved_load = pd.concat(unserved_load_dfs).query("UnservedLoadMW > 1e-3")

        # get timeseries info and calculate average unserved load per timeseries
        unserved_load["timeseries"] = unserved_load["TIMEPOINT"].map(tp_ts)
        unserved_load = (
            unserved_load.groupby("timeseries")["UnservedLoadMW"].mean().reset_index()
        )
        unserved_load = unserved_load.query("UnservedLoadMW > 0")
        if unserved_load.empty:
            # all load served
            ce_info["status"] = adequate
            continue

        candidates = unserved_load.loc[
            ~unserved_load["timeseries"].isin(ce_df["timeseries"]), :
        ]
        if candidates.empty:
            # some unserved load, but all timeseries with unserved load are already
            # in the model (unlikely)
            ce_info["status"] = stalled
            continue

        # find the timeries with the most unserved load that is not currently in the ce model
        ce_info["add_timeseries"] = candidates.loc[
            candidates["UnservedLoadMW"].idxmax(), "timeseries"
        ]
        ce_info["status"] = inadequate

    print("")

    # For each inadequate ce case, add the difficult timeseries to all the
    # input files for the ce case up to the maximum number of timeseries
    # allowed, then raise reserve margin incrementally, up to as high as
    # 100% if needed, but give up at that point.
    for ce_scen, ce_info in ce_scens.items():
        if ce_info["status"] not in {inadequate, stalled}:
            continue

        # if we've hit the limit for number of difficult-day cases in the RA
        # version of the CE model or already added all the days with unserved
        # load ("stalled" model), incrementally increase PRM level instead (but
        # no higher than options.max_prm_level)
        prm = read_ce_ra_in(ce_scen, "planning_reserve_margin.csv")
        if (
            ce_info["status"] == stalled
            or len(prm["TIMESERIES"].unique()) >= options.max_prm_timeseries_count
        ):
            if ce_info["status"] == stalled:
                print(
                    f"Scenario {ce_scen} has unserved load on a timeseries already included in the model."
                )
            else:
                print(
                    f"Reached maximum number of PRM timeseries ({options.max_prm_timeseries_count})."
                )

            if (prm["planning_reserve_margin"] >= options.max_prm_level).all():
                print(
                    f"WARNING: {ce_scen} model appears to be infeasible; there "
                    "is unserved load even with all planning reserve margins "
                    f"at the maximum level {options.max_prm_level:.6g}."
                )
                ce_info["status"] = infeasible
            else:
                # add 25% to PRM, or at least 1%, and round to nearest percent,
                # but don't exceed max allowed PRM level
                prm["planning_reserve_margin"] = (
                    (prm["planning_reserve_margin"] * 1.25)
                    .clip(prm["planning_reserve_margin"] + 0.01)
                    .round(2)
                    # final clip in case 1% would go above max level
                    .clip(upper=options.max_prm_level)
                )
                print(f"Raised PRM to {prm['planning_reserve_margin'].mean():.6g}.")
                write_ce_ra_in(ce_scen, "planning_reserve_margin.csv", prm)
                # convert "stalled" case to "inadequate" and try again with
                # higher PRM
                if ce_info["status"] == stalled:
                    ce_info["status"] = inadequate

            continue

        # haven't reached PRM series limit; add the worst timeseries to the ra
        # version of the ce model
        add_ts = ce_info["add_timeseries"]
        tag = ts_tag[add_ts]
        ra_scen = f"{ce_scen}_{tag}"  # same logic as ra_split_tag()
        print(f"Adding timeseries {add_ts} to scenario {ce_scen}.")

        # downsample timepoints for ce model and make zero-weight
        # this is similar to adjust/increase_timepoint_duration.py
        tp_dur = options.timepoint_duration
        for f in ["timeseries.csv", "timepoints.csv"]:
            ce_df = read_ce_ra_in(ce_scen, f)
            ra_df = read_ra_in(ra_scen, f).query("timeseries == @add_ts")
            if f == "timeseries.csv":
                assert (
                    ra_df["ts_duration_of_tp"].item() == 1
                ), "need to rewrite script to handle non-hourly RA models"
                ra_df["ts_duration_of_tp"] = tp_dur
                ra_df["ts_num_tps"] = ra_df["ts_num_tps"] // tp_dur
                ra_df["ts_scale_to_period"] = 0
            elif f == "timepoints.csv":
                # take every nth timepoint
                ra_df = ra_df.iloc[::tp_dur, :]
                # save for cross-referencing later
                add_tps = ra_df["timepoint_id"]
            write_ce_ra_in(ce_scen, f, pd.concat([ce_df, ra_df]))

        # copy matching rows for other timepoint files from ra model to
        # ra version of ce model
        for f, col in timepoint_files.items():
            ce_df = read_ce_ra_in(ce_scen, f)
            ra_df = read_ra_in(ra_scen, f).query(f"{col}.isin(@add_tps)")
            # write new version to the ce ra dir
            write_ce_ra_in(ce_scen, f, pd.concat([ce_df, ra_df]))

        # add the new timeseries to the PRM system
        f = "planning_reserve_margin.csv"
        ce_df = read_ce_ra_in(ce_scen, f)
        ra_df = pd.DataFrame(
            {
                "LOAD_ZONE": read_ra_in(ra_scen, "load_zones.csv")["LOAD_ZONE"],
                "TIMESERIES": add_ts,
                "planning_reserve_margin": options.initial_prm,
            }
        )
        write_ce_ra_in(ce_scen, f, pd.concat([ce_df, ra_df]))

    # report current status and save in ce model outputs dir (ra_status.txt)
    print("\nStatus of most recent model runs:")
    for ce_scen, ce_info in ce_scens.items():
        status = ce_info["status"]
        print(f"{ce_scen}: {status}")
        with open(Path(ce_scens[ce_scen]["outputs_dir"]) / "ra_status.txt", "w") as f:
            f.write(status)

    # generate new scenario files
    ce_scen_args = []
    for ce_scen, info in ce_scens.items():
        if info["status"] == inadequate:
            args = [
                f"--scenario-name",
                ce_scen,
                f"--inputs-dir",
                str(ce_ra_path(ce_scen)),
                f"--outputs-dir",
                info["outputs_dir"],
            ] + info["args"]
            ce_scen_args.append(shlex.join(args))

    ra_scen_args = []
    for ra_scen, info in ra_scens.items():
        if ce_scens[info["ce_scen"]]["status"] == inadequate:
            args = [
                f"--scenario-name",
                ra_scen,
                f"--inputs-dir",
                info["inputs_dir"],
                f"--outputs-dir",
                info["outputs_dir"],
            ] + info["args"]
            ra_scen_args.append(shlex.join(args))

    # create scenarios_build_ra.txt and scenarios_split_ra.txt
    new_files = []
    for scens_file, args in [
        (options.ce_scens_file, ce_scen_args),
        (options.ra_scens_file, ra_scen_args),
    ]:
        p = Path(scens_file)
        # copy to a _ra.txt file if not already such a file
        if p.stem.endswith("_ra"):
            next_scens_file = p
        else:
            next_scens_file = p.with_name(p.stem + "_ra" + p.suffix)
        with open(next_scens_file, "w") as f:
            f.writelines(a + "\n" for a in args)
        if args:
            new_files.append(next_scens_file)

    if new_files:
        print(f"\nCreated new RA scenario definitions.")
        print("\nNext, run the following commands:")
        for next_scens_file in new_files:
            print(f"switch solve-scenarios --scenario-list {next_scens_file}")
        print(
            f"python {sys.argv[0]} --ce-scens-file {new_files[0]} --ra-scens-file {new_files[1]}"
        )
    else:
        print("\nFinished iteration")
        print(
            "After adjustments, all capacity expansion scenarios are "
            "either adequate, stalled or infeasible."
        )


if __name__ == "__main__" and "ipykernel" not in sys.argv[0]:
    # running as a script, not from a jupyter environment
    main(get_script_args())
