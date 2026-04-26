"""
This script checks whether resource adequacy (RA) models found any unserved
load, and if so, adds the timeseries with the most unserved load into the
planning reserve margin timeseries of the capacity expansion (CE) models. It
also labels CE models as "adequate", "inadquate", or "infeasible".

This script is designed to be called iteratively. For the example below, the
main CE models are defined in `in/build` and saved in
`out/build/{scen1,scen2,...}`, the split/RA models are in `in/split`, the
definitions for the CE scenarios are in `in/build/scenarios_build.txt` and the
definitions for the RA split models are in `in/split/scenarios_split.txt`.

    # copy CE models to in/build/{scen1,scen2,...} and create initial versions of
    # in/build/scenarios_build.ra.txt and in/split/scenarios_split.ra.txt
    python ra_setup_iteration_models.py

    # iteratively solve CE and RA models
    while in/build/scenarios_build_ra.txt is not empty:
        switch solve-scenarios --scenario-list in/build/scenarios_build.ra.txt
        switch solve-scenarios --scenario-list in/resource_adequacy/scenarios_split.ra.txt
        python ra_add_difficult_timeseries.py in/build/scenarios_build.ra.txt in/resource_adequacy/scenarios_split.ra.txt

Additional notes:

At each iteration, this script overwrites the scenario list files with shorter
versions that contain only the scenarios that should be run for the next
iteration, i.e., still have unserved load.

After `--max-prm-timeseries-count` is reached, the PRM level will be multiplied
by 1.2 in each iteration until there is no unserved load or until
`--max-prm-level` is reached. If `--max-prm-level` is reached, the scenario will
be marked as "infeasible" in `out/build/scen?/ra_status.txt` and iteration will
stop.

If there is unserved load but only in timeseries that have already been added to
the CE model's PRM collection, then the script will raise the PRM level during
that iteration, as if `--max-prm-timeseries-count` had been reached. On the next
iteration, it will revert to adding difficult timeseries if there is still
unserved load. This generally should not occur.
"""

import argparse, shlex, shutil, sys, os, datetime, traceback
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
unknown = "unknown (incomplete RA results)"


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


def get_script_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "ce_scens_file",
        type=str,
        help="""
            Scenario list file used to define capacity expansion scenarios for
            resource adequacy iteration, e.g.,
            in/2030/s20x1/scenarios_build.ra..txt (created by
            ra_setup_iteration_models.py and rewritten by this script)
        """,
    )
    parser.add_argument(
        "ra_scens_file",
        type=str,
        help="""
            Scenario list file used to define resource adequacy (split)
            scenarios, e.g., in/2030/resource_adequacy/scenarios_split.ra.txt
            (created by ra_setup_iteration_models.py and rewritten by this
            script)
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

    def ra_split_tag(ra_scen):
        # parse ra split number (e.g., 0123) from an ra scenario name
        # (currently this is everything after the final underscore)
        # If this changes, also change f"{ce_scen}_{tag}" code below
        return ra_scen.rsplit("_", 1)[1]

    def read_ce_in(ce_scen, f):
        # read a file from the ce input directory
        # we turn off low_memory when reading to handle mixed column types in
        # timeseries.csv
        file = Path(ce_scens[ce_scen]["inputs_dir"]) / f
        return pd.read_csv(file, na_values=".", low_memory=False)

    def write_ce_in(ce_scen, f, df):
        # write a file in the ce input directory
        file = Path(ce_scens[ce_scen]["inputs_dir"]) / f
        df.to_csv(file, na_rep=".", index=False)
        # print(f"saved {file}.")

    def read_ra_in(ra_scen, f):
        # read an input file for an ra model from ra_main_dir / NNNN
        # note: if the lookup logic changes, also update tp_ts code below
        file = Path(ra_scens[ra_scen]["inputs_dir"]) / f
        return pd.read_csv(file, na_values=".")

    def read_ra_out(ra_scen, f):
        # read a ra-scen output
        file = Path(ra_scens[ra_scen]["outputs_dir"]) / f
        return pd.read_csv(file, na_values=".")

    t = datetime.datetime.now()
    print(f"Starting at {t.strftime('%Y-%m-%d %H:%M:%S')}.\n")

    # get start time as a sortable string to add to file names
    timestamp = t.strftime("%Y-%m-%d_%H-%M-%S")

    # cross-reference ce scenarios and ra scenarios
    for ra_scen, info in ra_scens.items():
        ce_scen = ra_scen.rsplit("_", 1)[0]
        info["ce_scen"] = ce_scen
        ce_scens[ce_scen].setdefault("ra_scens", []).append(ra_scen)

    # We could just report a message, but throwing an error is more likely to stop
    # an iterative loop, which should never reach here.
    if not ce_scens:
        raise RuntimeError(f"No scenarios are defined in {options.ce_scens_file}.")
    if not ra_scens:
        raise RuntimeError(f"No scenarios are defined in {options.ra_scens_file}.")

    if len(set(scen["inputs_dir"] for scen in ce_scens.values())) < len(ce_scens):
        # some scenarios share the same input directory, which will cause
        # trouble when we update the timeseries
        raise RuntimeError(
            f"Some capacity expansion models in {options.ce_scens_file} share the "
            "same inputs directory. This script must be run with unique directories "
            "for each model. Hint: you should use the .ra.txt scenario lists created "
            "by ra_setup_iteration_models.py."
        )

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
        # get list of ra scens in this ce scen
        rs = ce_scens[ce_scen]["ra_scens"]

        # get average unserved load for each timeseries
        # read in all unserved load files, getting total for each timepoint across all zones
        unreadable_cases = []
        unserved_load_dfs = [
            pd.DataFrame(
                columns=["TIMEPOINT", "UnservedLoadMW", "UnservedLoad_GWh_typical_year"]
            )
        ]
        n_scens = len(rs)
        print(f"Reading unserved load files for {ce_scen}:")
        for i, r in enumerate(rs):
            try:
                df = (
                    read_ra_out(r, "unserved_load.csv")
                    .groupby("TIMEPOINT")[
                        ["UnservedLoadMW", "UnservedLoad_GWh_typical_year"]
                    ]
                    .sum()
                    .reset_index()
                )
                unserved_load_dfs.append(df)
            except (FileNotFoundError, TimeoutError):
                unreadable_cases.append(r)
            if i == 0 or (i + 1) % (n_scens / 10) < 1:
                print(f"  Read {i+1}/{n_scens} files ({(i+1)/n_scens:.0%}).")

        if unreadable_cases:
            print(
                f"WARNING: Unable to read unserved load for {len(unreadable_cases)} RA cases for {ce_scen}."
            )

        print()

        unserved_load_tp = pd.concat(unserved_load_dfs).query("UnservedLoadMW > 1e-3")

        # save peak coincident unserved load and total unserved energy
        if unserved_load_tp.empty:
            uns_mw = 0
            uns_gwh = 0
        else:
            uns_mw = unserved_load_tp["UnservedLoadMW"].max()
            uns_gwh = unserved_load_tp["UnservedLoad_GWh_typical_year"].sum()
        ce_info["peak_unserved_load_mw"] = uns_mw
        ce_info["total_unserved_energy_gwh"] = uns_gwh

        # get timeseries info and calculate average unserved load per timeseries
        unserved_load_tp["timeseries"] = unserved_load_tp["TIMEPOINT"].map(tp_ts)

        # stash to save later
        ce_info["unserved_load_df"] = unserved_load_tp

        unserved_load = (
            unserved_load_tp.groupby("timeseries")["UnservedLoadMW"]
            .mean()
            .reset_index()
            .query("UnservedLoadMW > 0")
        )

        if unserved_load.empty:
            if unreadable_cases:
                # no unserved load, but not all cases could be read
                # try again without changing the model
                ce_info["status"] = unknown
            else:
                # all load served
                ce_info["status"] = adequate
            continue

        ce_prm_timeseries = (
            read_ce_in(ce_scen, "planning_reserve_margin.csv")["TIMESERIES"]
            .unique()
            .astype(str)
        )
        candidates = unserved_load.loc[
            ~(unserved_load["timeseries"].astype("str") + "_prm").isin(
                ce_prm_timeseries
            ),
            :,
        ]
        if candidates.empty:
            if unreadable_cases:
                # some unserved load, but only on timeseries that are already in
                # the reserve set; however there may be some unreadable timeseries
                # with unserved load, so status is unknown
                ce_info["status"] = unknown
            else:
                # There is some unserved load, but all timeseries with unserved
                # load are already in the model reserve set (unlikely)
                ce_info["status"] = stalled
            continue

        # find the timeries with the most unserved load that is not currently in the ce model
        ce_info["add_timeseries"] = candidates.loc[
            candidates["UnservedLoadMW"].idxmax(), "timeseries"
        ]
        ce_info["ts_unserved_load_mw"] = candidates["UnservedLoadMW"].max()
        ce_info["status"] = inadequate

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
        prm = read_ce_in(ce_scen, "planning_reserve_margin.csv")
        if (
            ce_info["status"] == stalled
            or len(prm["TIMESERIES"].unique()) >= options.max_prm_timeseries_count
        ):
            if ce_info["status"] == stalled:
                print(
                    f"Scenario {ce_scen} has unserved load, but only on timeseries "
                    "that are already included in the PRM set for the model."
                )
            else:
                print(
                    "Reached maximum number of PRM timeseries "
                    f"({options.max_prm_timeseries_count})."
                )

            if (prm["planning_reserve_margin"] >= options.max_prm_level).all():
                print(
                    f"WARNING: {ce_scen} model is infeasible; there is unserved "
                    "load even with all planning reserve margins at the maximum "
                    f"level {options.max_prm_level:.6g}."
                )
                ce_info["status"] = infeasible
            else:
                # # add 25% to PRM, or at least 1%, and round to nearest percent,
                # # but don't exceed max allowed PRM level
                # prm["planning_reserve_margin"] = (
                #     (prm["planning_reserve_margin"] * 1.25)
                #     .clip(prm["planning_reserve_margin"] + 0.01)
                #     .round(2)
                #     # final clip in case 1% would go above max level
                #     .clip(upper=options.max_prm_level)
                # )
                # add 20% to PRM but don't exceed limit
                prm["planning_reserve_margin"] = 1.2
                prm["planning_reserve_margin"] = (
                    prm["planning_reserve_margin"] * 1.2
                ).clip(upper=options.max_prm_level)
                print(f"Raised PRM to {prm['planning_reserve_margin'].mean():.6g}.\n")
                write_ce_in(ce_scen, "planning_reserve_margin.csv", prm)
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
        print(
            f"Adding timeseries {add_ts} with unserved load "
            f"{ce_info['ts_unserved_load_mw']:.6g} MWa to scenario {ce_scen}."
        )
        print()

        def add_prm_rows(ce_df, ra_df, time_cols):
            """
            Add "_prm" suffix to any timepoint or timeseries columns in the RA
            dataframe to make the RA rows distinct from standard timepoints and
            timeseries, then combine the CE rows and RA rows into one dataframe.
            """
            ce_df = ce_df.copy()
            ra_df = ra_df.copy()
            if isinstance(time_cols, str):
                time_cols = [time_cols]
            for col in time_cols:
                if col in ce_df:
                    ce_df[col] = ce_df[col].astype(str)
                if col in ra_df:
                    ra_df[col] = ra_df[col].astype(str) + "_prm"
            return pd.concat([ce_df, ra_df], ignore_index=True)

        # downsample timepoints for ce model and make zero-weight
        # this is similar to adjust/increase_timepoint_duration.py
        tp_dur = options.timepoint_duration
        for f in ["timeseries.csv", "timepoints.csv"]:
            ce_df = read_ce_in(ce_scen, f)
            ra_df = read_ra_in(ra_scen, f).query("timeseries == @add_ts")
            if f == "timeseries.csv":
                assert (
                    ra_df["ts_duration_of_tp"].item() == 1
                ), "need to rewrite script to handle non-hourly RA models"
                ra_df["ts_duration_of_tp"] = tp_dur
                ra_df["ts_num_tps"] = ra_df["ts_num_tps"] // tp_dur
                ra_df["ts_scale_to_period"] = 0
                time_cols = "timeseries"
            elif f == "timepoints.csv":
                # take every nth timepoint
                ra_df = ra_df.iloc[::tp_dur, :]
                time_cols = ["timepoint_id", "timestamp", "timeseries", "tp_date"]
                # save for cross-referencing later
                add_tps = ra_df["timepoint_id"]  # used in query below
            write_ce_in(ce_scen, f, add_prm_rows(ce_df, ra_df, time_cols))

        # copy matching rows for other timepoint files from ra model to
        # ce model
        for f, col in timepoint_files.items():
            ce_df = read_ce_in(ce_scen, f)
            ra_df = read_ra_in(ra_scen, f).query(f"{col}.isin(@add_tps)")
            # write new version to the ce ra dir
            write_ce_in(ce_scen, f, add_prm_rows(ce_df, ra_df, col))

        # add the new timeseries to the PRM system
        f = "planning_reserve_margin.csv"
        ce_df = read_ce_in(ce_scen, f)
        ra_df = pd.DataFrame(
            {
                "LOAD_ZONE": read_ra_in(ra_scen, "load_zones.csv")["LOAD_ZONE"],
                "TIMESERIES": add_ts,
                "planning_reserve_margin": options.initial_prm,
            }
        )
        write_ce_in(ce_scen, f, add_prm_rows(ce_df, ra_df, "TIMESERIES"))

    # report current status and save in ce model outputs dir (ra_status.txt)
    print("Status of most recent model runs:")
    for ce_scen, ce_info in ce_scens.items():
        status = ce_info["status"]
        print(
            f"{ce_scen}: {status}, unserved load: "
            f"{ce_info['peak_unserved_load_mw']:.6g} MW "
            f"/ {ce_info['total_unserved_energy_gwh']:.6g} GWh"
        )
        out_dir = Path(ce_scens[ce_scen]["outputs_dir"])
        iter_dir = out_dir / f"iter_{timestamp}"
        with open(out_dir / "ra_status.txt", "w") as f:
            f.write(status + "\n")
        try:
            # attempt to create iteration snapshots, but don't worry if it fails
            iter_dir.mkdir(exist_ok=True)
            status_df = pd.DataFrame(
                {
                    k: [ce_info.get(k, None)]
                    for k in [
                        "status",
                        "peak_unserved_load_mw",
                        "total_unserved_energy_gwh",
                        "add_timeseries",
                        "ts_unserved_load_mw",
                    ]
                }
            )
            with open(out_dir / "total_cost.txt") as f:
                status_df["ce_model_cost"] = float(f.read().strip())
            status_df["timestamp"] = timestamp
            status_df.to_csv(iter_dir / "status.csv", index=False)
            ce_info["unserved_load_df"].to_csv(
                iter_dir / "unserved_load.csv", index=False
            )
            for f in ["gen_cap.csv", "cost_components.csv"]:
                if (out_dir / f).is_file():
                    shutil.copy2(out_dir / f, iter_dir / f)
            # print(f"Saved iteration snapshot in {iter_dir}.")
        except Exception as err:
            print(
                f"Error saving iteration snapshot for {ce_scen}: {traceback.format_exc(limit=0).strip()}"
            )

    # re-write scenarios_build.ra.txt and scenarios_split.ra.txt with only the
    # remaining scenarios
    new_files = []
    scen_info = [  # scens_file, scens, status function
        (
            options.ce_scens_file,
            ce_scens,
            lambda info: info["status"],
        ),
        (
            options.ra_scens_file,
            ra_scens,
            lambda info: ce_scens[info["ce_scen"]]["status"],
        ),
    ]
    for scens_file, scens, status in scen_info:
        args = [
            [
                f"--scenario-name",
                info["scenario_name"],
                f"--inputs-dir",
                info["inputs_dir"],
                f"--outputs-dir",
                info["outputs_dir"],
            ]
            + info["args"]
            for info in scens.values()
            if status(info) not in {adequate, infeasible}
        ]

        with open(scens_file, "w") as f:
            f.writelines(shlex.join(a) + "\n" for a in args)
        if args:
            new_files.append(scens_file)

    if new_files:
        print(f"\nCreated new RA scenario definitions.")
        print("\nNext, run the following commands:")
        for scens_file in new_files:
            print(f"switch solve-scenarios --scenario-list {scens_file}")
        print(f"python {sys.argv[0]} {' '.join(new_files)}")
    else:
        print("\nFinished iteration")
        print(
            "After adjustments, all capacity expansion scenarios are "
            "either adequate, stalled or infeasible."
        )


if __name__ == "__main__" and "ipykernel" not in sys.argv[0]:
    # running as a script, not from a jupyter environment
    main(get_script_args())
