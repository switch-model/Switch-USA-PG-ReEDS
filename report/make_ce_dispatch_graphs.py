# %% setup

from pathlib import Path

import pandas as pd
import numpy as np

# avoid problems with pyplot on Jupyter
import matplotlib

# fix rcparams import problem with incompatible jupyter/matplotlib versions
# matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams


from reporting_info import (
    assert_all_in,
    zone_groups,
    scen_names,
    gen_tech_names,
    ce_in_dir,
    ce_out_dir,
    supply_colors,
    demand_patterns,
    supply_cols,
    demand_cols,
    summary_out_dir,
)

out_dir = summary_out_dir / "ce_dispatch_graphs"

# use EI standard font
rcParams["font.family"] = "Montserrat"
# Turn off minus signs being replaced with boxes
# rcParams["axes.unicode_minus"] = False


def read_in(scen, file, **kwargs):
    return pd.read_csv(ce_in_dir / file, na_values=".", **kwargs)


def read_out(scen, file, **kwargs):
    return pd.read_csv(ce_out_dir / scen / file, na_values=".", **kwargs)


def append(df_list, cat, mw):
    df_list.append(pd.DataFrame({"cat": cat, "mw": mw}))


def assert_all_in(grp1, grp2, msg="missing values"):
    # raise an error if any members of grp1 are not in grp2
    missing = set(grp1) - set(grp2)
    if missing:
        raise AssertionError(f"msg: {missing}")


# %% Read and aggregate data

# identify firm sources for min/max net load assessment
firm_supplies = {
    "Dist. Solar": False,
    "Nuclear": True,
    "Coal": True,
    "Onshore Wind": False,
    "Large Solar": False,
    "Imports": False,
    "Gas CT": True,
    "Gas CCGT": True,
    "Storage": True,
    "Offshore Wind": False,
    "Hydro": True,
    "Other": True,
}
assert_all_in(supply_cols, firm_supplies, "missing graph labels in firm_supplies")
assert_all_in(firm_supplies, supply_cols, "unexpected graph labels firm_supplies")

scen_data = {scen: {} for scen in scen_names}
for scen in scen_data:
    supply_dfs = []
    demand_dfs = []
    dispatch = read_out(scen, "dispatch.csv")
    cols = {
        "gen_load_zone": "load_zone",
        "timestamp": "timestamp",
        "cat": "cat",
        "DispatchGen_MW": "mw",
    }
    supply_dfs.append(
        dispatch.assign(cat=lambda g: g["gen_tech"].map(gen_tech_names))
        .rename(columns=cols)[cols.values()]
        .groupby(["load_zone", "timestamp", "cat"])
        .agg({"mw": "sum"})
        .reset_index("cat")
    )
    tp_weight = dispatch.groupby("timestamp")["tp_weight_in_year_hrs"].mean().to_dict()
    # del dispatch

    tp_timestamp = read_in(scen, "timepoints.csv").set_index("timepoint_id")[
        "timestamp"
    ]

    zone_balance = read_out(scen, "load_balance.csv").set_index(
        ["load_zone", "timestamp"]
    )
    dist_balance = read_out(scen, "local_td_energy_balance_wide.csv").set_index(
        ["load_zone", "timestamp"]
    )
    transmission = read_out(scen, "transmission.csv")

    # customer demand, including flex loads
    for c in ["ShiftDemand", "EEDemandReduction"]:
        if c not in dist_balance:
            dist_balance[c] = 0

    # note: loads are negative in dist_balance but positive in zone_balance
    append(demand_dfs, cat="Base Demand", mw=-dist_balance["zone_demand_mw"])
    # append(demand_dfs, cat="PRM", mw=-dist_balance["planning_reserves"])
    append(demand_dfs, cat="Modified Demand", mw=-dist_balance["ShiftDemand"])
    append(demand_dfs, cat="Modified Demand", mw=-dist_balance["EEDemandReduction"])

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

    # transmission nominal flows and losses
    # we report these separately so that when we aggregate, the
    # nominal flows will cancel out as needed but the losses will stay
    # (the other way would be to just report net imports, but then
    # when summed for the whole country, transmission losses will look
    # like exports and it will be confusing)
    DispatchTx = read_out(scen, "DispatchTx.csv").rename(
        columns={
            "TRANS_TIMEPOINTS_1": "lz1",
            "TRANS_TIMEPOINTS_2": "lz2",
            "TRANS_TIMEPOINTS_3": "timepoint",
        }
    )
    DispatchTx["timestamp"] = DispatchTx["timepoint"].map(tp_timestamp)

    # get net trans flows to this zone (zone = lz2) minus flows from this zone (zone
    # = lz1) ignoring losses
    net_imports = (
        DispatchTx.groupby(["lz2", "timestamp"])["DispatchTx"].sum()
        - DispatchTx.groupby(["lz1", "timestamp"])["DispatchTx"].sum()
    )
    net_imports.index.names = ["load_zone", "timestamp"]
    # ideal imports minus actual net imports = losses (we assign to importing
    # zone because that's easiest)
    # These have some small negative (actually zero) values so we round a bit
    tx_losses = (net_imports - zone_balance["TXPowerNet"]).round(1)

    append(supply_dfs, cat="Imports", mw=net_imports)
    append(demand_dfs, cat="Base Demand", mw=tx_losses)

    # del zone_balance, dist_balance, transmission, DispatchTx, net_imports, tx_losses

    timestamp_ts = read_in(scen, "timepoints.csv").set_index("timestamp")["timeseries"]

    # combine into single dfs, removing duplicate categories
    for sd, dfs in {"supply": supply_dfs, "demand": demand_dfs}.items():
        df = (
            pd.concat(dfs)
            .groupby(["load_zone", "timestamp", "cat"])
            .sum()
            .reset_index()
        )
        df["timeseries"] = df["timestamp"].map(timestamp_ts)
        # scen_data['high_fossil']['supply']
        scen_data[scen][sd] = df

# del supply_dfs, demand_dfs

# %% aggregate across regions, select key days and graph results
# zone_data: scen_data: day: data
region_data = {zg: {scen: {} for scen in scen_data} for zg in zone_groups}
for region, rdata in region_data.items():
    for scen, sdata in rdata.items():
        # create data_tp df with supply and demand for every timepoint for
        # this set of zones; this is in wide format with one column per cat
        dfs = {
            sd: (
                df[df["load_zone"].isin(zone_groups[region])]
                .groupby(["timeseries", "timestamp", "cat"])["mw"]
                .sum()
                .unstack("cat")
            )
            for sd, df in scen_data[scen].items()
        }
        # shift negative imports to exports (or base demand)
        supply = dfs["supply"]
        demand = dfs["demand"]
        # demand["Exports"] = (-supply["Imports"]).clip(lower=0)
        demand["Base Demand"] += (-supply["Imports"]).clip(lower=0)
        supply["Imports"] = supply["Imports"].clip(lower=0)

        # pre-sum demand data for display in front of stacked area supply chart,
        # then combine with supply data
        assert_all_in(demand_cols, demand.columns, "missing columns in demand df")
        assert_all_in(demand.columns, demand_cols, "unexpected columns in demand df")
        for i in range(1, len(demand_cols)):
            demand[demand_cols[i]] += demand[demand_cols[i - 1]]
        data_tp = pd.concat([supply, demand], axis=1)

        # find peak, valley and average timeseries for each zone group, omitting PRM days
        firm_cols = [c for c, firm in firm_supplies.items() if firm and c in data_tp]
        firm_supply = data_tp.loc[
            ~data_tp.index.get_level_values("timestamp")
            .astype(str)
            .str.endswith("_prm"),
            firm_cols,
        ].sum(axis=1)
        peak_ts = firm_supply.idxmax()[0]
        valley_ts = firm_supply.idxmin()[0]

        # store graph data for each zone group, for peak, valley and average timeseries
        sdata["peak"] = data_tp.xs(peak_ts, level="timeseries")
        sdata["valley"] = data_tp.xs(valley_ts, level="timeseries")

        weight = (
            data_tp.index.get_level_values("timestamp").map(tp_weight).rename("weight")
        )
        hour_of_day = (
            data_tp.index.get_level_values("timestamp")
            .astype(str)
            .str[9:11]
            .astype(int)
            .rename("hour_of_day")
        )

        weighted = (
            data_tp.mul(weight.values, axis=0)
            .groupby(hour_of_day)
            .sum()
            .div(weight.to_series().groupby(hour_of_day).sum().values, axis=0)
        )
        sdata["average"] = weighted


# %% Prepare panel plot
# create output dir if needed
out_dir.mkdir(parents=True, exist_ok=True)

for region, rdata in region_data.items():
    n_rows = len(rdata)
    n_cols = max(len(ddata) for ddata in rdata.values())

    fig, panel = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(3 * n_cols, 4.5),
        sharey=True,
        sharex=True,
    )

    # make an n_rows x n_cols array of axes (panel will be just a list
    # or single axes if n_rows or n_cols = 1)
    if n_rows == 1 and n_cols == 1:
        all_axes = np.array([[panel]])
    else:
        all_axes = panel.reshape((n_rows, n_cols))

    for scen, ax_row in zip(rdata, all_axes):
        for day_type, ax in zip(rdata[scen], ax_row):
            sub = rdata[scen][day_type].copy()
            if sub.index.name == "timestamp":
                # note: timestamps are YYYYRMMDDhh, where R is the repetition
                # number, starting with 2007=0
                s = str(sub.index[0])
                d_label = f"like {int(s[5:7])}/{int(s[7:9])}/{7+int(s[4]):02d}"
                label = f"{day_type} day ({d_label})"
                sub.index = (
                    sub.index.astype(str).str[9:11].astype(int).rename("hour_of_day")
                )
            elif sub.index.name == "hour_of_day":
                d_label = "average"
                label = f"{day_type} day"

            # wrap back to the start for graphing
            sub.loc[24, :] = sub.loc[0, :]

            # convert hour of day to a proper timestamp
            times = pd.to_datetime(sub.index, unit="h")
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
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
            ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
            ax.tick_params(axis="x", which="major", length=7)
            ax.tick_params(axis="x", which="minor", length=4)
            # ax.grid(which='major', axis='x')

        # label leftmost axes on this row
        ax = ax_row[0]
        ax.set_ylabel(f"GW")
        ax.text(
            -0.55,  # x & y relative to graph size
            0.5,
            scen.replace("_", "\n"),
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
        )

    # # label bottom axes
    # for ax in all_axes[-1]:
    #     ax.set_xlabel("Time of Day")

    # move plots apart a little vertically (fraction of axes "h"eight)
    fig.subplots_adjust(hspace=0.55)

    # Place legend on the top-rightmost plot, with net load at top and positive labels reversed to match plot
    # legend_ord = ["Customer Load"] + list(reversed(pos_cols)) + neg_cols
    handles, labels = ax.get_legend_handles_labels()
    ord = list(range(len(labels) - 1, -1, -1))
    trans_labels = labels

    all_axes[0][-1].legend(
        [handles[o] for o in ord],
        [trans_labels[o] for o in ord],
        bbox_to_anchor=(1.02, 1.03),
        loc="upper left",
        facecolor="none",
        edgecolor="none",
    )
    # all_axes[0][-1].legend(
    #     bbox_to_anchor=(1.02, 1.03),
    #     loc="upper left",
    #     facecolor="none",
    #     edgecolor="none",
    # )
    # plt.show()
    for fmt in ["png", "pdf"]:
        out_file = f"{out_dir}/{region.replace('.', '_')}_daily.{fmt}"
        fig.savefig(
            out_file,
            bbox_inches="tight",
            pad_inches=0.05,
            transparent=True,
            dpi=300,
        )
    print(f"saved {out_file}")

# %%
