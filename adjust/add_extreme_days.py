# %% setup
import sys
import argparse
import shutil
import csv
from pathlib import Path
import pandas as pd

# files to process and columns to extend
# from grep -rl 480 switch/in/test_imports_no_retire/2024/s4 | xargs -I{} sh -c 'echo; echo $(basename "{}"); head -n 1 {}'
timepoint_files = {
    # "hydro_timepoints.csv": "timepoint_id",  # not used for this project
    "loads.csv": "TIMEPOINT",
    "variable_capacity_factors.csv": "TIMEPOINT",
    "water_node_tp_flows.csv": "TIMEPOINTS",
}
# from grep -rl 2024_p141_0 switch/in/test_imports_no_retire/2024/s4 | xargs -I{} sh -c 'echo; echo $(basename "{}"); head -n 1 {}'
timestamp_files = {
    "graph_timestamp_map.csv": "timestamp",
}
# from grep -rl 2024_p141 switch/in/test_imports_no_retire/2024/s4 | xargs -I{} sh -c 'echo; echo $(basename "{}"); head -n 1 {}'
timeseries_files = {
    # "hydro_timepoints.csv": "tp_to_hts",  # sometimes reuses timeseries ID, but not used for this project
    "graph_timestamp_map.csv": "timeseries",
    # "hydro_timeseries.csv": "timeseries",   # REAM version, not for this project
}


def parse_arguments():
    # process command line options (name of inputs dir and )
    parser = argparse.ArgumentParser()
    parser.add_argument("in_dir", help="Path to the input directory")
    # parser.add_argument(
    #     "--peak-timeseries",
    #     action="store_const",
    #     const="peak",
    #     dest="timeseries",
    #     default="peak",
    #     help="Add extra copies of peak timeseries with 0 weight(default)",
    # )
    # parser.add_argument(
    #     "--all-timeseries",
    #     action="store_const",
    #     const="all",
    #     dest="timeseries",
    #     help="Add extra copies of all timeseries with 0 weight",
    # )
    options = parser.parse_args()
    return options


# %% main code


# for testing:
# sys.argv = ['script', 'switch/in/test_imports_no_retire/2024/s4']
def main():
    options = parse_arguments()
    in_dir = Path(options.in_dir)
    out_dir = in_dir.with_name(in_dir.stem + "_prm")

    def read(f):
        return pd.read_csv(in_dir / f, na_values=".")

    def write(df, file):
        # timepoints and timeseries may now appear to contain mixed string and
        # int, so we quote all non-numeric columns to make sure they go through
        # to Switch as strings (pyomo will cheerfully read these as mixed
        # integer/string sets, but that can cause trouble later). But we also
        # have to take extra steps to avoid quoting the headers, which confuses
        # pyomo. (note: pyomo converts quoted numbers to numbers too, so this
        # doesn't help.)
        # with open(out_dir / file, "w", newline="") as f:
        #     f.write(",".join(df.columns) + "\n")
        #     df.to_csv(
        #         f, na_rep=".", index=False, header=False, quoting=csv.QUOTE_NONNUMERIC
        #     )
        df.to_csv(out_dir / file, na_rep=".", index=False)
        print(f"saved {out_dir / file}.")

    print(f"Copying {in_dir} to {out_dir}.")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(in_dir, out_dir)

    # define planning reserve margins and determine whether we are interested
    # in peak days or all days
    prr = read("planning_reserve_requirements.csv")
    prz = read("planning_reserve_requirement_zones.csv")
    enforcement_timescales = prr["prr_enforcement_timescale"].unique()
    assert (
        len(enforcement_timescales) == 1
    ), "planning_reserve_requirement_zones.csv must only have one value for prr_enforcement_timescale"
    enforcement_timescale = enforcement_timescales[0]
    assert enforcement_timescale in {
        "all_timepoints",
        "peak_load",
    }, "prr_enforcement_timescale in planning_reserve_requirement_zones.csv must be 'all_timepoints' or 'peak_load'"

    # create timeseries and timepoints tables with duplicated versions of the
    # relevant timepoints; the timepoint, timestamp and timeseries columns will be
    # converted to str and the duplicated ones will be tagged with "_res" at the end.
    # We also keep the original versions as 'old_timepoint', 'old_timeseries' and
    # "old_timestamp" for lookups when needed.

    timeseries = read("timeseries.csv")
    timepoints = read("timepoints.csv")
    ts_cols = timeseries.columns.to_list()
    tp_cols = timepoints.columns.to_list()
    timepoints = timepoints.rename(columns={"timepoint_id": "timepoint"}).merge(
        timeseries
    )

    print(f"Updating {out_dir} with extreme-day timeseries ({enforcement_timescale}).")

    if enforcement_timescale == "peak_load":
        # find peak hour each period
        load = (
            read("loads.csv").groupby("TIMEPOINT")["zone_demand_mw"].sum().reset_index()
        )
        load = load.merge(timepoints, left_on="TIMEPOINT", right_on="timepoint")
        peak_idx = load.groupby("ts_period")["zone_demand_mw"].idxmax()
        peak = load.loc[peak_idx, :]
        new_tp = peak[["timeseries"]].merge(timepoints)
    else:
        # duplicate all timepoints
        new_tp = timepoints.copy()

    # set weights to 0 for the new version of these timepoints
    new_tp["ts_scale_to_period"] = 0

    # create old_x columns in new_tp and timepoints, convert standard
    # columns to str and add "_res" to the new_tp version
    for col in ["timepoint", "timestamp", "timeseries"]:
        for df in new_tp, timepoints:
            df["old_" + col] = df[col]
            df[col] = df[col].astype(str)
        new_tp[col] += "_prm"

    # add new timepoints and save
    timepoints = pd.concat([timepoints, new_tp], ignore_index=True)
    # temporary hack: only keep prm timepoints
    # timepoints = pd.concat([new_tp], ignore_index=True)
    timeseries = timepoints[ts_cols + ["old_timeseries"]].drop_duplicates()

    write(
        timepoints.rename(columns={"timepoint": "timepoint_id"})[tp_cols],
        "timepoints.csv",
    )
    write(timeseries[ts_cols], "timeseries.csv")

    df_name = {id(timepoints): "timepoints", id(timeseries): "timeseries"}

    # now add copies to the corresponding files as needed
    update_files = [
        (timepoint_files, timepoints, "timepoint"),
        (timestamp_files, timepoints, "timestamp"),
        (timeseries_files, timeseries, "timeseries"),
    ]
    for i, (cols, ref_df, ref_col) in enumerate(update_files):
        # Go through each group of column updates, looking up the
        # new value for that column and duplicating rows if needed.
        # Then for any other columns in the same table that need updating,
        # assign the correct value to match the updated column.
        # If a table appears in multiple groups, process it in the first
        # one and then skip it in later groups.
        for file, col in cols.items():
            if any(file in grp[0] for grp in update_files[:i]):
                # appeared in a previous update group; skip
                continue
            # lookup new time column and duplicate rows as needed
            df = read(file)
            # rename col to "old" version, match to old_ref_col and bring in
            # ref_col as col. Keep new cols from ref_df when there is a conflict
            # (useful for referring to them in the peek-ahead below if needed.)
            df_new = df.rename(columns={col: "old_" + col}).merge(
                ref_df.rename(columns={ref_col: col}),
                left_on="old_" + col,
                right_on="old_" + ref_col,
                suffixes=("_drop", ""),
            )
            # peek into future groups and update other columns if needed
            for _cols, _ref_df, _ref_col in update_files[i + 1 :]:
                if file in _cols:
                    # Look up the new value for this column.
                    # note: the groups are ordered so that if we get here, df_new
                    # will already have merged with timepoints, so we just need to
                    # retrieve the relevant extra column from there.
                    # e.g., df_new['tp_to_hts'] = df_new['timeseries']
                    df_new[_cols[file]] = df_new[_ref_col]

            # print(
            #     f"saving {file}, {col=}, {i=}, ref_df={df_name[id(ref_df)]}, {ref_col=}"
            # )
            write(df_new[df.columns], file)

    # create planning reserves file (zone, timeseries, margin) from new_tp and prr_whatnot
    zone_prm = (
        prr.rename(columns={"prr_cap_reserve_margin": "planning_reserve_margin"})
        .merge(prz)
        .groupby("LOAD_ZONE")["planning_reserve_margin"]
        .sum()  # should only be one margin per zone, but we sum just in case
        .reset_index()
    )
    # replicate across all extreme-day timeseries
    zone_ts_prm = pd.concat(
        [zone_prm.assign(TIMESERIES=ts) for ts in new_tp["timeseries"].unique()]
    )[["LOAD_ZONE", "TIMESERIES", "planning_reserve_margin"]]
    write(zone_ts_prm, "planning_reserve_margin.csv")


if __name__ == "__main__" and "ipykernel" not in sys.argv[0]:
    # running as a script, not from a jupyter environment
    main()

# %%
