# %% setup

print("loading libraries")
import os, datetime, argparse, shlex, shutil, pickle, io, textwrap, re
from pathlib import Path

import pandas as pd
import numpy as np

# avoid problems with pyplot on Jupyter
import matplotlib

# fix rcparams import problem with incompatible jupyter/matplotlib versions
# matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg

from reporting_info import (
    assert_all_in,
    rto_groups,
    region_info,
    scen_names,
    gen_tech_names,
    ce_in_dir,
    ra_in_dir,
    ce_out_dir,
    ra_out_dir,
    summary_out_dir,
    fonts_dir,
    logo_file,
    logo_zoom,
    timepoint_files,
    supply_colors,
    demand_patterns,
    supply_cols,
    demand_cols,
)

out_dir = summary_out_dir / "weekly_dispatch_graphs"

print("loaded libraries")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--reuse-extreme-days",
    action="store_true",
    help="Reuse cached extreme-day selections from dispatch_graph_key_dates.pkl.",
)
# for testing, drop the jupyter arg
# import sys; sys.argv = sys.argv[:1]
args = parser.parse_args()

# graph (name, filter, stat) definitions
graph_defs = [
    [
        ("Summer high net load day", "season == 'Summer'", "pct_gap"),
        ("Winter high net load day", "season == 'Winter'", "pct_gap"),
    ],
    [
        ("Median day", None, "-(abs(load_pct - 0.5) + abs(re_pct - 0.5))"),
        ("Low net load day", None, "-pct_gap"),
    ],
]

ra_scens_file = ra_in_dir / "scenarios_split.txt"


# register all user-supplied fonts (user-supplied)
if fonts_dir.is_dir():
    for ext in ["ttf", "otf"]:
        for f in fonts_dir.glob(f"*.{ext}"):
            matplotlib.font_manager.fontManager.addfont(f)

# use EI standard font:
# Montserrat for reports or Century Gothic for presentations
fig_font = "Montserrat"

matplotlib.rcParams.update(
    {
        "font.family": fig_font,
        # don't convert text to outlines in SVG; computer importing svg and creating
        # final PDF will need the font installed
        "svg.fonttype": "none",
    }
)

# Turn off minus signs being replaced with boxes (with some fonts)
# matplotlib.rcParams["axes.unicode_minus"] = False


def read_ce_in(scen, file, **kwargs):
    return pd.read_csv(ce_in_dir / file, na_values=".", **kwargs)


def read_ce_out(scen, file, **kwargs):
    return pd.read_csv(ce_out_dir / scen / file, na_values=".", **kwargs)


def read_ra_in(scen, file, **kwargs):
    return pd.read_csv(ra_in_dir / file, na_values=".", **kwargs)


def read_ra_out(scen, file, **kwargs):
    # concat all the files matching this name from the numbered subdirs of the
    # RA output dir
    df = pd.concat(
        (
            pd.read_csv(f, na_values=".", **kwargs)
            for f in (ra_out_dir / scen).glob(f"*/{file}")
        )
    )
    return df


def copy_file(src, dest_dir):
    src = Path(src)
    # copy, resolving symlinks if needed
    shutil.copy2(src.resolve(), Path(dest_dir) / src.name)


def append(df_list, cat, mw):
    df_list.append(pd.DataFrame({"cat": cat, "mw": mw}))


def get_scenario_args(scenario_file):
    parser = argparse.ArgumentParser()
    # process/consume these arguments if present
    parser.add_argument("--scenario-name")
    parser.add_argument("--inputs-dir")
    parser.add_argument("--outputs-dir")
    parser.add_argument("--skip-generic-output", action="store_true")
    parser.add_argument(
        "--skip-output-file",
        "--skip-output-files",
        dest="skip_output_files",
        nargs="+",
        default=[],
        action="extend",
    )

    with open(scenario_file) as f:
        scenario_strings = f.read().splitlines()

    scen_info = {}
    for scen_str in scenario_strings:
        if scen_str.startswith("#"):
            continue
        options, other_args = parser.parse_known_args(shlex.split(scen_str))
        scen_info[options.scenario_name] = vars(options) | {"args": other_args}

    return scen_info


def date_to_week(dt):
    """Convert Pandas date timestamp into the date at the start of that week."""
    return dt - pd.to_timedelta((dt.dayofweek + 1) % 7, unit="D")


def date_to_label(dt):
    """Convert Pandas date timestamp to a text representation."""
    return dt.strftime("%Y-%m-%d")


def graphing_out_dir(scen, date):
    if isinstance(date, str):
        # already a label
        week_label = date
    else:
        # convert to start of the week if needed, then to a label
        week_label = date_to_label(date_to_week(date))
    return ra_out_dir / "weekly_graphing" / scen / week_label


def graphing_in_dir(scen, date):
    if isinstance(date, str):
        # already a label
        week_label = date
    else:
        # convert to start of the week if needed, then to a label
        week_label = date_to_label(date_to_week(date))
    return ra_in_dir / "weekly_graphing" / scen / week_label


def read_graphing_out(scen, date, file, **kwargs):
    return pd.read_csv(graphing_out_dir(scen, date) / file, na_values=".", **kwargs)


def read_graphing_in(scen, date, file, **kwargs):
    return pd.read_csv(graphing_in_dir(scen, date) / file, na_values=".", **kwargs)


# %% Scan the daily models for key stats

# there are daily timeseries, which run 1/1/2007 - 12/31/2013,
# but omit the last day of leap years
ts_date = pd.DataFrame(
    {
        "timeseries": read_ra_in(scen_names[0], "timeseries.csv")["timeseries"],
        "hist_date": [
            d
            for d in pd.date_range(start="2007-01-01", end="2013-12-31")
            if not d.dayofyear == 366
        ],
    }
).set_index("timeseries")["hist_date"]
date_ts = ts_date.reset_index().set_index("hist_date")["timeseries"]

tp_hour = pd.DataFrame(
    {
        "timepoint": read_ra_in(scen_names[0], "timepoints.csv")["timepoint_id"],
        "hist_hour": [
            h
            for h in pd.date_range(
                start="2007-01-01 00:00:00", end="2013-12-31 23:00:00", freq="1H"
            )
            if not h.dayofyear == 366
        ],
    }
).set_index("timepoint")["hist_hour"]

ts_case = pd.concat(
    (
        pd.read_csv(f, na_values=".").assign(case=f.parent.name)
        for f in (ra_in_dir).glob(f"*/timeseries.csv")
    )
).set_index("timeseries")["case"]

zg_cache_file = Path("dispatch_graph_key_dates.pkl")
zg_key_days = None
if args.reuse_extreme_days and zg_cache_file.exists():
    print(f"Using extreme days cached in {zg_cache_file}")
    with zg_cache_file.open("rb") as f:
        zg_key_days_read = pickle.load(f)
    # keep only those needed for current selection (and reset
    # if not available)
    zg_key_days = {}
    try:
        for scen in scen_names:
            zg_key_days[scen] = {}
            for g in rto_groups:
                zg_key_days[scen][g] = zg_key_days_read[scen][g]
    except KeyError:
        print(f"No record for {scen}:{g} in {zg_cache_file}; re-scanning.")  # type: ignore
        zg_key_days = None

if zg_key_days is None:
    # regenerate zg_key_days
    zg_key_days = {}

    for scen in scen_names:
        print(f"finding extreme days for {scen} scenario")
        all_ts_dispatch = (
            read_ra_out(scen, "zonal_total_dispatch.csv")
            .groupby(["load_zone", "timeseries"])[["variable_gen_dispatch"]]
            .mean()
            .reset_index()
        )

        tp_ts = read_ra_in(scen, "timepoints.csv").set_index("timepoint_id")[
            "timeseries"
        ]
        loads = (
            read_ra_in(scen, "loads.csv")
            .rename(columns={"LOAD_ZONE": "load_zone", "TIMEPOINT": "timepoint"})
            .assign(timeseries=lambda df: df["timepoint"].map(tp_ts))
            .groupby(["load_zone", "timeseries"])["zone_demand_mw"]
            .mean()
            .reset_index()
        )

        all_ts = all_ts_dispatch.merge(loads, on=["load_zone", "timeseries"])
        all_ts["hist_date"] = all_ts["timeseries"].map(ts_date)
        all_ts["season"] = np.where(
            all_ts["hist_date"].dt.month.between(5, 10), "Summer", "Winter"
        )

        # zg = 'CAISO'; zones = zone_groups[zg]
        for zg, zones in zone_groups.items():
            group_data = (
                all_ts.drop(columns="timeseries")
                .query("load_zone.isin(@zones)")
                .groupby(["season", "hist_date"])
                .sum()
                .reset_index()
            )
            # find load and renewable energy percentiles and gap (across all
            # years, all seasons)
            group_data[["load_pct", "re_pct"]] = group_data[
                ["zone_demand_mw", "variable_gen_dispatch"]
            ].rank(pct=True)
            group_data["pct_gap"] = group_data["load_pct"] - group_data["re_pct"]

            # collect extreme dates based on graph definitions
            for row in graph_defs:
                for name, filter, stat in row:
                    if not filter:
                        filter = "index == index"
                    extreme_idx = group_data.query(filter).eval(stat).idxmax()
                    zg_key_days.setdefault(scen, {}).setdefault(zg, {})[name] = (
                        group_data.loc[extreme_idx, ["hist_date", "load_pct", "re_pct"]]
                        .rename({"hist_date": "date"})
                        .to_dict()
                    )

    with zg_cache_file.open("wb") as f:
        pickle.dump(zg_key_days, f)


# identify the full week for each group
needed_weeks = sorted(
    {
        (scen, date_to_week(day_data["date"]))
        for scen, scen_data in zg_key_days.items()
        for zg_data in scen_data.values()
        for day_data in zg_data.values()
    }
)


####
# construct weekly models by merging daily models and create a
# scenario list for them. Then let user run them, come back
# and run this script again.

to_run = []
for scen, week_stamp in needed_weeks:
    if not (graphing_out_dir(scen, week_stamp) / "total_cost.txt").exists():
        to_run.append((scen, week_stamp))

if to_run:
    all_ra_scens = get_scenario_args(ra_scens_file)
    scens_to_run = []
    for scen, week_stamp in to_run:
        week_label = date_to_label(week_stamp)
        # identify all the ones in the right weeks, probably based on
        # date -> timeseries -> daily scenario mapping
        dates_to_merge = pd.date_range(start=week_stamp, periods=7).to_series()
        ts_to_merge = dates_to_merge.map(date_ts).dropna()
        cases_to_merge = ts_to_merge.map(ts_case)
        first_case = cases_to_merge.iloc[0]

        # merge all the daily cases in this week together into a single weekly case
        # first, copy the whole dir, updating any symlinks there
        weekly_dir = graphing_in_dir(scen, week_stamp)
        print(f"creating {weekly_dir}")
        weekly_dir.mkdir(parents=True, exist_ok=True)
        for f in (ra_in_dir / first_case).iterdir():
            if f.is_file():
                copy_file(f, weekly_dir)
        # now merge the time-based files and save to the weekly dir
        read_merged_df = lambda f: pd.concat(
            read_ra_in(scen, Path(case) / f) for case in cases_to_merge
        )
        write_weekly_df = lambda df, f: df.to_csv(weekly_dir / f, index=False)
        for f in timepoint_files.keys():
            df = read_merged_df(f)
            write_weekly_df(df, f)
        weekly_tp = read_merged_df("timepoints.csv")
        weekly_tp["timeseries"] = week_label
        write_weekly_df(weekly_tp, "timepoints.csv")
        weekly_ts = read_merged_df("timeseries.csv").iloc[[0], :]
        weekly_ts["timeseries"] = week_label
        n_days = len(weekly_tp) / weekly_ts["ts_num_tps"].item()
        weekly_ts["ts_num_tps"] *= n_days
        weekly_ts["ts_scale_to_period"] /= n_days
        write_weekly_df(weekly_ts, "timeseries.csv")

        # store info for the scenarios file (based on the first daily case in the week)
        ra_scen = f"{scen}_{first_case}"
        scen_info = all_ra_scens[ra_scen]
        scens_to_run.append(
            [
                "--scenario-name",
                f"{scen}_{week_label}",
                "--inputs-dir",
                str(graphing_in_dir(scen, week_stamp)),
                "--outputs-dir",
                str(graphing_out_dir(scen, week_stamp)),
            ]
            + scen_info["args"]
            # get full reporting
            + ["--exclude-module", "study_modules.reduce_reporting"]
        )

    graphing_scen_file = ra_in_dir / "scenarios_graphing.txt"
    with open(graphing_scen_file, "w") as f:
        f.writelines(shlex.join(a) + "\n" for a in scens_to_run)
    print(f"Created scenario list `{graphing_scen_file}`.")

    print(
        "Weekly models need to be run to prepare for graphing. "
        "Please run switch solve-scenarios --scenario-list "
        f"{graphing_scen_file} "
        "and then run this script again."
    )
    exit(0)
else:
    print("All necessary scenarios have been solved. Preparing graphs.")

# %% Read and aggregate weekly data

# zg_key_days: scen: zg: name: date/load_pct/re_pct
print("Collecting dispatch data for weeks with extreme days in all regions.")
for scen, scen_data in zg_key_days.items():
    for zg, zg_data in scen_data.items():
        for day_type, date_data in zg_data.items():
            date = date_data["date"]

            supply_dfs = []
            demand_dfs = []

            tp_timestamp = read_graphing_in(scen, date, "timepoints.csv").set_index(
                "timepoint_id"
            )["timestamp"]

            cols = {
                "gen_load_zone": "load_zone",
                "timestamp": "timestamp",
                "cat": "cat",
                "DispatchGen_MW": "mw",
            }

            dispatch_by_gen = read_graphing_out(scen, date, "dispatch.csv")
            assert_all_in(
                dispatch_by_gen["gen_tech"].unique(),
                gen_tech_names.keys(),
                "unexpected gen_tech values in dispatch.csv",
            )
            dispatch = (
                dispatch_by_gen.assign(cat=lambda g: g["gen_tech"].map(gen_tech_names))
                .rename(columns=cols)[cols.values()]
                .groupby(["load_zone", "timestamp", "cat"])
                .agg({"mw": "sum"})
                .reset_index("cat")
            )
            del dispatch_by_gen
            supply_dfs.append(dispatch)

            zone_balance = read_graphing_out(scen, date, "load_balance.csv").set_index(
                ["load_zone", "timestamp"]
            )
            dist_balance = read_graphing_out(
                scen, date, "local_td_energy_balance_wide.csv"
            ).set_index(["load_zone", "timestamp"])

            # customer demand, including flex loads
            for c in ["ShiftDemand", "EEDemandReduction"]:
                # make sure these columns exist, as zero if necessary
                if c not in dist_balance:
                    dist_balance[c] = 0

            # note: loads are negative in dist_balance but positive in zone_balance
            append(demand_dfs, cat="Base Demand", mw=-dist_balance["zone_demand_mw"])
            append(demand_dfs, cat="Modified Demand", mw=-dist_balance["ShiftDemand"])
            append(
                demand_dfs,
                cat="Modified Demand",
                mw=-dist_balance["EEDemandReduction"],
            )

            # note: the categories below are lumped into more general groups for simpler graphing
            # local T&D losses
            append(
                demand_dfs,
                cat="Base Demand",
                mw=zone_balance["WithdrawFromCentralGrid"]
                - dist_balance["InjectIntoDistributedGrid"],
            )
            # storage charging
            append(
                demand_dfs,
                cat="Modified Demand",
                mw=zone_balance["ZoneTotalStorageCharging"],
            )

            # Adjust for simultaneous charging and discharging, which can
            # happen during curtailment, since it has no cost and may be
            # simpler for the solver (subtract any simultaneous charging
            # from both supply and demand)
            charge_discharge = pd.DataFrame(
                {
                    "charging": zone_balance["ZoneTotalStorageCharging"],
                    "discharging": dispatch.query("cat=='Storage'")["mw"],
                }
            )
            charge_discharge["simultaneous"] = charge_discharge.min(axis=1)
            append(supply_dfs, cat="Storage", mw=-charge_discharge["simultaneous"])
            append(
                demand_dfs,
                cat="Modified Demand",
                mw=-charge_discharge["simultaneous"],
            )

            # transmission nominal flows and losses
            # we report these separately so that when we aggregate, the
            # nominal flows will cancel out as needed but the losses will stay
            # (the other way would be to just report net imports, but then
            # when summed for the whole country, transmission losses will look
            # like exports and it will be confusing)
            DispatchTx = read_graphing_out(scen, date, "DispatchTx.csv").rename(
                columns={
                    "TRANS_TIMEPOINTS_1": "lz1",
                    "TRANS_TIMEPOINTS_2": "lz2",
                    "TRANS_TIMEPOINTS_3": "timepoint",
                }
            )
            DispatchTx["timestamp"] = DispatchTx["timepoint"].map(tp_timestamp)

            # get net trans flows to this zone (zone = lz2) minus flows from this zone (zone
            # = lz1) ignoring losses
            ideal_net_imports = (
                DispatchTx.groupby(["lz2", "timestamp"])["DispatchTx"].sum()
                - DispatchTx.groupby(["lz1", "timestamp"])["DispatchTx"].sum()
            )
            ideal_net_imports.index.names = ["load_zone", "timestamp"]
            # ideal imports minus actual net imports = losses (we assign to importing
            # zone because that's easiest)
            # These have some small negative (actually zero) values so we round a bit
            tx_losses = (ideal_net_imports - zone_balance["TXPowerNet"]).round(1)

            append(supply_dfs, cat="Imports", mw=ideal_net_imports)
            append(demand_dfs, cat="Base Demand", mw=tx_losses)

            # # diagnostics
            # sd_dfs = []
            # for sd, dfs in {"supply": supply_dfs, "demand": demand_dfs}.items():
            #     df = (
            #         pd.concat(dfs).loc[("p10", 20302060200), :]
            #         # .query("load_zone == 'p10' and timestamp == 20302060200")
            #     )
            #     sd_dfs.append(df)
            # supply, demand = sd_dfs
            # print(f'{supply["mw"].sum()=}, {demand["mw"].sum()=}')

            # combine into single dfs, removing duplicate categories
            sd_dfs = []
            for sd, dfs in {"supply": supply_dfs, "demand": demand_dfs}.items():
                df = (
                    pd.concat(dfs)
                    .groupby(["load_zone", "timestamp", "cat"])
                    .sum()
                    .reset_index()
                )
                # aggregate for this region (zg)
                df = (
                    df[df["load_zone"].isin(zone_groups[zg])]
                    .groupby(["timestamp", "cat"])["mw"]
                    .sum()
                    .unstack("cat")
                    .reset_index()
                )
                sd_dfs.append(df)

            supply, demand = sd_dfs
            demand = demand.drop(columns="timestamp")

            # shift negative imports to exports (or base demand)
            demand["Base Demand"] += (-supply["Imports"]).clip(lower=0)
            supply["Imports"] = supply["Imports"].clip(lower=0)

            # pre-sum demand data for display in front of stacked area supply chart,
            # then combine with supply data
            assert_all_in(demand_cols, demand.columns, "missing columns in demand df")
            assert_all_in(
                demand.columns, demand_cols, "unexpected columns in demand df"
            )
            for i in range(1, len(demand_cols)):
                demand[demand_cols[i]] += demand[demand_cols[i - 1]]
            data_tp = pd.concat([supply, demand], axis=1)
            # add timestamps for the matching historical hours
            # (some of these will match date_data["hist_date"])
            data_tp["hist_hour"] = data_tp["timestamp"].map(tp_hour)
            date_data["data"] = data_tp


# %% Prepare panel plot
# test matplotlib configuration
try:
    # make sure matplotlib_inline is compatible with matplotlib
    # (otherwise this script can crash in jupyter environments
    # later when functions in plt load matplotlib-inline, if
    # matplotlib <= 3.6 and pyplot >= 0.2.2 (maybe other versions)
    # (matplotlib 3.6 seems to work fine with matplotlib-inline 0.1.7)
    fig = plt.figure()
except Exception as e:
    print(
        "Unable to create matplotlib figure. Sometimes this happens in Jupyter "
        "environments if matplotlib-inline is too new for matplotlib itself."
    )
    raise


# Create a date-formatting function that will show the date as mm/dd, and prefix
# it with a star if it matches mark_date.
def date_star_formatter(mark_date):
    def formatter(x, pos):
        label = mdates.num2date(x).strftime("%m/%d")
        if mdates.num2date(x).date() == mark_date.date():
            # mark this date as the key one
            label = f"({label})"
        return label

    return formatter


def selected_date_formatter(dates):
    to_label = {dt.date() for dt in dates}

    def formatter(x, pos):
        dt = mdates.num2date(x).date()
        if dt in to_label:
            return dt.strftime("%m/%d")
        else:
            return ""

    return formatter


def patch_matplotlib_svg_font_name(svg_path):
    s = Path(svg_path).read_text(encoding="utf-8")
    # Convert Matplotlib-style CSS font shorthand like:
    #   font: 12px 'Montserrat';
    #   font: 700 12px 'Montserrat';
    # to:
    #   font-family: 'Montserrat'; font-size: 12px;
    #   font-family: 'Montserrat'; font-size: 12px; font-weight: 700;

    # This helps SVG importers such as Affinity Designer and MS Word that may
    # ignore the CSS font shorthand.

    def repl(m):
        weight = m.group("weight")
        size = m.group("size")
        family = m.group("family")
        weight_name = (
            "bold"
            if isinstance(weight, str) and weight.isdigit() and int(weight) >= 700
            else weight
        )
        if weight_name == "bold":
            # Change the font-family to work with Affinity Designer and MS Word
            family += " Bold"
        parts = [
            f"font-family: '{family}';",
            f"font-size: {size}px;",
        ]
        if weight:
            parts.append(f"font-weight: {weight};")
        return " ".join(parts)

    s = re.sub(
        r"font:\s*(?:(?P<weight>[1-9]00)\s+)?(?P<size>[0-9.]+)px\s+'(?P<family>[^']+)'\s*;",
        repl,
        s,
    )

    Path(svg_path).write_text(s, encoding="utf-8")


# create output dir if needed
out_dir.mkdir(parents=True, exist_ok=True)

print("Generating panel plots")
for scen, scen_data in zg_key_days.items():
    for zg, zg_data in scen_data.items():
        n_rows = len(graph_defs)
        n_cols = max(len(row) for row in graph_defs)

        fig, panel = plt.subplots(
            nrows=n_rows, ncols=n_cols, figsize=(3 * n_cols, 3), sharey=True
        )
        fig.suptitle(f"{zg}, {scen.replace('_', ' ')}", fontweight="bold", y=1.04)
        # make an n_rows x n_cols array of axes (panel will be just a list
        # or single axes if n_rows or n_cols = 1)
        if n_rows == 1 and n_cols == 1:
            all_axes = np.array([[panel]])
        else:
            all_axes = panel.reshape((n_rows, n_cols))

        for row, ax_row in zip(graph_defs, all_axes):
            for (day_type, filter, stat), ax in zip(row, ax_row):
                day_data = zg_data[day_type]
                date = day_data["date"]
                sub = day_data["data"].copy()

                # d_label = f"like {date.strftime('%Y')}"
                # label = f"{day_type} day ({d_label})"
                label = day_type

                # create a dummy hour at 0:00 the next day for graphing
                # (by wrapping back to the start, which is how Switch represents it)
                wrap_day = sub.loc[[0], :]
                hh = sub["hist_hour"].iloc[-2:].to_list()
                wrap_day["hist_hour"] = hh[1] + (hh[1] - hh[0])
                sub = pd.concat([sub, wrap_day])

                times = sub["hist_hour"]
                s_colors = {k: v for k, v in supply_colors.items() if k in sub}
                s_cols = list(s_colors.keys())

                # plot supply
                ax.stackplot(
                    times,
                    sub[s_cols].values.T * 0.001,
                    labels=s_cols,
                    colors=s_colors.values(),
                    # edgecolor="gray",
                    # linewidth=0.25,
                )

                # plot loads as lines
                cols = [c for c in demand_cols if c != "PRM"]
                for col in cols:
                    ax.plot(
                        times,
                        sub[col] * 0.001,
                        label=col,
                        zorder=10,
                        **demand_patterns[col],
                    )

                ax.set_title(label)
                # round edges of box to nearest exact date
                ax.set_xlim([round(x, 0) for x in ax.get_xlim()])
                ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0]))
                ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[12]))
                ax.xaxis.set_major_formatter(mticker.NullFormatter())
                ax.xaxis.set_minor_formatter(mticker.NullFormatter())
                # show dates on minor ticks, so they are centered each day
                # ax.xaxis.set_minor_formatter(
                #     selected_date_formatter(
                #         [times.iloc[0], times.iloc[-2]]
                #     )  # last one skips the extra hour
                # )
                # ax.xaxis.set_minor_formatter(mdates.DateFormatter("%m/%d"))
                ax.tick_params(axis="x", which="major", length=7)
                ax.tick_params(axis="x", which="minor", length=4)  # , labelrotation=45)
                # ax.grid(which='major', axis='x')

                x_labels = [
                    (0, times.iloc[0].strftime("%m/%d")),
                    (1, times.iloc[-1].strftime("%m/%d")),
                ]
                for pos, label in x_labels:
                    ax.text(
                        pos,  # x & y relative to graph size
                        -0.15,
                        label,
                        transform=ax.transAxes,
                        ha="center",
                        va="top",
                        fontsize=10,
                    )

                ax.axvspan(
                    date,
                    date + pd.Timedelta(hours=24),
                    facecolor="none",
                    # alpha=0.2,
                    edgecolor="black",
                    zorder=100,
                    linewidth=0.5,
                )
                ax.axvspan(
                    date,
                    date + pd.Timedelta(hours=24),
                    facecolor="mediumpurple",
                    alpha=0.35,
                    edgecolor="none",
                    zorder=0,
                )

            # label leftmost axes on this row
            ax = ax_row[0]
            ax.set_ylabel(f"GW")

        # # label bottom axes
        # for ax in all_axes[-1]:
        #     ax.set_xlabel("Time of Day")

        # move plots apart a little vertically (fraction of axes "h"eight)
        fig.subplots_adjust(hspace=0.6)

        # Place legend on the top-rightmost plot, with net load at top and
        # positive labels reversed to match plot
        # legend_ord = ["Customer Load"] + list(reversed(pos_cols)) + neg_cols
        handles, labels = ax.get_legend_handles_labels()
        ord = list(range(len(labels) - 1, -1, -1))

        legend = all_axes[0][-1].legend(
            [handles[o] for o in ord],
            [labels[o] for o in ord],
            bbox_to_anchor=(1.1, 1.4),
            loc="upper left",
            facecolor="none",
            edgecolor="none",
        )

        # add logo if file is present
        if logo_file.exists():
            # find right edge of legend in figure coords
            fig.canvas.draw()  # pin down legend position
            legend_bbox = legend.get_window_extent(
                renderer=fig.canvas.get_renderer()
            ).transformed(fig.transFigure.inverted())
            ab = AnnotationBbox(
                OffsetImage(mpimg.imread(logo_file), zoom=logo_zoom),
                xy=(legend_bbox.x1 - 0.02, 0.03),
                xycoords=fig.transFigure,  # use figure coords
                box_alignment=(1, 0),  # right, bottom
                frameon=False,
                pad=0,
            )
            fig.add_artist(ab)

        for fmt in ["pdf", "svg"]:  # "png", "pdf", "svg"]:
            file = str(out_dir / f"{zg.replace('.', '')} {scen}.{fmt}")
            fig.savefig(
                file,
                bbox_inches="tight",
                pad_inches=0.05,
                transparent=True,
                dpi=300,
            )
            if fmt == "svg":
                patch_matplotlib_svg_font_name(file)
            print(f"saved {file}.")

        # close figure to save memory
        plt.close(fig)

# %%
