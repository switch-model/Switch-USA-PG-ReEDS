"""
calculate mean, quartiles and range of these values
- per existing/new build and technology, across all projects (weighted by gen_capacity_limit_mw or existing capacity if applicable, otherwise just one item per zone (thermal generators)):
    - 2030 capital cost
    - gen_connect_cost_per_mw
    - fixed O&M
    - variable O&M
    - capacity factor
- per fuel, across all zones (unweighted)
    - 2030 cost per MMBtu
- total generator capacity (existing) or limit (2030) (MW) of each type
- total coincident load curve (all zones)

Describe the RPS/CES/min-build policies in place
"""

# %% setup data
import sys
import pandas as pd
import numpy as np
from duckdb import query as q
from pathlib import Path

# in_dir = "test"
# in_dir = "test_eia_2023_860m_2025"
# in_dir = "test_eia_2024_860m_2025"
# in_dir = "test_gem"
# in_dir = "test_flex"

# for testing: sys.argv[1] = "in/test_flex/2024/s4"
# for testing: sys.argv[1] = "switch/in/test/2030/p1"

idir = Path(sys.argv[1])
try:
    tag_start = idir.parts.index("in") + 1
except ValueError:
    tag_start = 0
label = "_".join(idir.parts[tag_start:])

out_html = label + ".html"


def icsv(f):
    return pd.read_csv(idir / f, na_values=".")


print("Processing data files")

info = icsv("gen_info.csv")
build = icsv("gen_build_predetermined.csv")

# calculate existing cap and max possible cap
# use that to calculate existing vs. possible additions
# calculate stats above separately for possible additions vs. existing

# drop existing plants that will retire before the start of the study
retire_year = build["build_year"] + build["GENERATION_PROJECT"].map(
    info.set_index("GENERATION_PROJECT")["gen_max_age"]
)
# assume GenX-style retirement: must survive beyond end of first period
build = build[retire_year > icsv("periods.csv")["period_end"].min()]

existing_cap = build.groupby("GENERATION_PROJECT")["build_gen_predetermined"].sum()
info["existing_cap"] = info["GENERATION_PROJECT"].map(existing_cap).fillna(0.0)
info["new_cap"] = (
    info["gen_capacity_limit_mw"] - info["existing_cap"]
)  # keep gen_cap_limit NAs
# gather capital and O&M costs for future build years (if capacity isn't predetermined)
# NOTE: for multi-period models, this will create new-build rows for each study period
future_costs = icsv("gen_build_costs.csv").merge(
    icsv("periods.csv"), left_on=["BUILD_YEAR"], right_on="INVESTMENT_PERIOD"
)
future_costs = future_costs.merge(
    build,
    how="left",
    left_on=["GENERATION_PROJECT", "BUILD_YEAR"],
    right_on=["GENERATION_PROJECT", "build_year"],
).query(
    "build_gen_predetermined.isna()"  # drop any that found a match (predetermined cap)
)[
    [
        "GENERATION_PROJECT",
        "BUILD_YEAR",
        "gen_overnight_cost",
        "gen_fixed_om",
        "gen_storage_energy_overnight_cost",
    ]
]
info = info.merge(future_costs, how="left", on="GENERATION_PROJECT")
# zero out future additions for any that don't have future construction costs
# (this is how Switch identifies buildable projects)
info.loc[info["gen_overnight_cost"].isna(), "new_cap"] = 0.0

# treat missing gen_storage_energy_overnight_cost as zero for storage projects
info.loc[
    info["gen_storage_efficiency"].notna()
    & info["gen_storage_energy_overnight_cost"].isna(),
    "gen_storage_energy_overnight_cost",
] = 0.0

# check for any existing projects that can be extended in the future
# (generally none from PowerGenome)
new_and_old = info.query("(existing_cap > 0) and (new_cap.isna() | new_cap > 0)").loc[
    :, ["GENERATION_PROJECT", "gen_tech", "existing_cap", "new_cap"]
]
assert new_and_old.empty, "Some existing projects can also be extended in the future."

# calculate and apply capacity factors
# (assume no weighting for now)
cf = icsv("variable_capacity_factors.csv")
cf_map = cf.groupby("GENERATION_PROJECT")["gen_max_capacity_factor"].mean()
info["annual_cap_factor"] = info["GENERATION_PROJECT"].map(cf_map)
del cf, cf_map  # save some memory

# split into existing and new build rows, doubling up if needed (unlikely)
info = pd.concat(
    [
        info.query("existing_cap > 0").assign(
            status="existing", capacity=info["existing_cap"]
        ),
        info.query("new_cap.isna() | (new_cap > 0)").assign(
            status="new", capacity=lambda df: df["new_cap"]
        ),
    ],
    ignore_index=True,
)

# Identify groups for plotting
info.loc[info["status"] == "existing", "group"] = "Existing " + info["gen_tech"]
info.loc[info["status"] == "new", "group"] = (
    info["gen_tech"] + " (" + info["BUILD_YEAR"].astype("Int64").astype(str) + ")"
)
info["group"] = info["group"].astype(str).str.replace("_", " ")

fuel = icsv("fuel_cost.csv")

# %% make plots
print("Preparing graphs")

import textwrap
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from statsmodels.stats.weightstats import DescrStatsW

info["group_wrapped"] = info["group"].map(
    lambda x: "<br>".join(textwrap.wrap(x, width=30))
)

# assign colors
energy_colors = {
    "storage": "red",
    "biomass": "limegreen",
    "wind": "deepskyblue",
    "water": "blue",
    "coal": "black",
    "naturalgas": "gray",
    "uranium": "yellow",
    "geothermal": "seagreen",
    "sun": "gold",
    "demand_response": "purple",
    "imports": "magenta",
}

info["color"] = info["gen_energy_source"].map(energy_colors)
missing = info.query("color.isna()")["gen_energy_source"].unique().astype(str)
if len(missing) > 0:
    print(
        f"The following energy sources don't have colors assigned: {', '.join(missing)}"
    )


def group_stats(df, col):
    # make dataframe with columns x (=group), lowerfence, q1, median, q3 and
    # upperfence
    def gs(df):
        vals = df[col]
        if vals.empty:
            # use 0 for empty dataframe to avoid errors (e.g. in non-new-build cases)
            vals = pd.Series([0.0], name=vals.name)
        d = DescrStatsW(vals, weights=df["capacity"].fillna(1))
        q25, q50, q75 = d.quantile([0.25, 0.5, 0.75])
        return pd.Series(
            {
                "lowerfence": d.data.min(),
                "q1": q25,
                "median": q50,
                "q3": q75,
                "upperfence": d.data.max(),
                # "marker_color": df["color"].iloc[0], # doesn't work
            }
        )

    active_rows = df.loc[df[col].notna(), :].copy()
    if active_rows.empty:
        # PROBLEM: if a group (or maybe the whole df?) is empty, .apply can
        # generate the columns, but pandas just returns the original columns
        # instead of dummy columns from .apply. So instead, we just return None
        # as a sign that there is nothing to plot.
        return None
    else:
        agg_df = (
            active_rows.groupby("group_wrapped")[[col, "capacity", "color"]]
            .apply(gs)
            .reset_index()
            .rename(columns={"group_wrapped": "x"})
        )
        return agg_df


def make_tech_fig(col, value_name, value_units):
    """
    Return a plotly fig object showing values from the specified column,
    grouped by year and technology, with a title like f"{value_name} by Technology"
    """
    stats = group_stats(info, col)
    if stats is None:
        # no data, make empty plot
        box = go.Box()
        fig = go.Figure(data=[box])
        fig.add_annotation(
            text="No data",
            xref="x",
            yref="y",
            x=0,
            y=0.5,
            showarrow=False,
            font={"size": 20},
        )
        fig.update_xaxes(range=[-1, 1])
        fig.update_yaxes(range=[0, 1])
    else:
        # normal plot
        box = go.Box(boxpoints=False, **stats)
        fig = go.Figure(data=[box])

    fig.update_layout(
        title=f"{value_name} by Technology",
        xaxis_title="Technology",
        yaxis_title=f"{value_name} ({value_units})",
        # xaxis_tickangle=90,
    )
    return fig


tech_plots = [
    ("gen_overnight_cost", "Gen Capital Cost", "2024$/MW"),
    ("gen_storage_energy_overnight_cost", "Gen Storage Capital Cost", "2024$/MWh"),
    ("gen_connect_cost_per_mw", "Gen Connection Capital Cost", "2024$/MW"),
    ("gen_variable_om", "Gen Variable O&M", "2024$/MWh"),
    ("gen_fixed_om", "Gen Fixed O&M", "2024$/MW-year"),
    ("annual_cap_factor", "VRE Capacity Factor", "fraction"),
]

figs = []

figs.extend(make_tech_fig(*args) for args in tech_plots)

# per fuel, across all zones (unweighted): 2030 cost per MMBtu
fuel["color"] = fuel["fuel"].map(energy_colors)
fuel_fig = px.box(
    fuel,
    x="fuel",
    y="fuel_cost",
    # color="color", # treated as category, not color label
    color="fuel",
    facet_col="period",
    points="outliers",  # or "all", False, etc.
    title="Fuel Cost by Fuel Type",
)
fuel_fig.update_layout(
    yaxis_title="Fuel Cost (2024$/MMBtu)", xaxis_title="Fuel", boxmode="group"
)
figs.append(fuel_fig)

total_cap = (
    info.dropna(subset=["capacity"])
    .groupby("group_wrapped")["capacity"]
    .sum()
    .reset_index()
    .rename(columns={"group_wrapped": "technology"})
)
total_cap["technology"] = total_cap["technology"].str.replace(
    "(2030)", "(2030 limit)", regex=False
)

# total generator capacity (existing) or limit (2030) (MW) of each type
cap_fig = px.bar(
    total_cap,
    x="technology",
    y="capacity",
    title="Capacity by Technology",
)
cap_fig.update_layout(
    yaxis_title="Existing Capacity or Limit (MW)", xaxis_title="Technology"
)
cap_fig.update_yaxes(type="log")
figs.append(cap_fig)

# total coincident load curve (all zones)
# (ignoring sampling and periods for now)
load = (
    icsv("loads.csv")
    .groupby("TIMEPOINT")["zone_demand_mw"]
    .sum()
    .reset_index()
    .rename(columns={"zone_demand_mw": "load"})
    .sort_values(by="load", axis=0, ascending=False)
    .reset_index(drop=True)
) * 0.001  # convert to GW
total_load = int(load["load"].mean() * 8760 * 0.001)  # in TWh
load_fig = px.line(
    load,
    y="load",
    title=f"2030 U.S. Load Duration Curve (total = {total_load} TWh/y)",
)
load_fig.update_layout(
    yaxis_title="U.S. Coincident Load (GW)", xaxis_title=f"Sample Hour"
)
load_fig.update_yaxes(range=[0, None])
figs.append(load_fig)

print("Preparing capex map")
# make a map of capex by ReEDS region and insert after capital cost graph
import json
import os

try:
    # avoid warnings about using Shapely vs. PyGEOS if both are installed
    import shapely

    os.environ["USE_PYGEOS"] = "0"
except:
    pass

import geopandas as gpd

# has columns WKT, rb (zone); filter to US zones only and convert to geometry and load_zone
# a shapefile with more info (state, RTO, interconnect and major region) is at
# https://github.com/NREL/ReEDS-2.0/tree/main/inputs/shapefiles/US_PCA
df = pd.read_csv(
    "https://raw.githubusercontent.com/NREL/ReEDS-2.0/refs/heads/main/inputs/shapefiles/US_CAN_MEX_PCA_polygons.csv"
)
df = df[df["rb"].str[1:].astype(int) <= 134]
zones = (
    gpd.GeoDataFrame(
        df,
        geometry=gpd.GeoSeries.from_wkt(df["WKT"]),
        crs="EPSG:4326",
    )
    .rename(columns={"rb": "load_zone"})
    .drop(columns="WKT")
    .set_index("load_zone", drop=False)
)
col = "CCGT capex ($/kW)"
zones[col] = (
    zones["load_zone"]
    .map(
        info.query("gen_tech == 'NaturalGas_1-on-1 Combined Cycle (H-Frame)_Moderate'")
        .groupby("gen_load_zone")["gen_overnight_cost"]
        .mean()
        * 0.001
    )
    .round()
)

# # Albers projected version (no basemap)
# zones = zones.to_crs(4326)  # WGS 84 (lat/lon) for plotly
# fig = px.choropleth(
#     zones,
#     geojson=json.loads(zones.to_json()),
#     locations=zones.index,
#     # featureidkey="properties.rb",
#     color=capex_col,
#     projection="albers usa"
# )
# fig.update_geos(
#     showcountries=True,
#     showcoastlines=True,
#     showland=True,
#     fitbounds="locations"
# )

# Polygons
fig = px.choropleth_map(
    zones,
    geojson=json.loads(zones.to_json()),
    locations=zones.index,  # unique id per feature
    color=col,
    # opacity=0.5,
    title=f"{col} by model zone",
)
fig.update_traces(marker_line_width=0.5, marker_line_color="white")

# Labels at representative points
# centers = zones.geometry.representative_point()
# zones["label_lon"] = centers.x
# zones["label_lat"] = centers.y
# fig.add_trace(
#     go.Scattermap(
#         lon=zones["label_lon"],
#         lat=zones["label_lat"],
#         mode="text",
#         text=zones["load_zone"],
#         textfont=dict(size=12),
#         hoverinfo="text",
#     )
# )

# Center/zoom
minx, miny, maxx, maxy = zones.total_bounds
fig.update_layout(
    # map_style="open-street-map",  # no token needed
    map_zoom=3.3,  # exact level depends on browser width
    map_center={"lat": (miny + maxy) / 2, "lon": (minx + maxx) / 2},
    margin=dict(l=0, r=0, t=0, b=0),
)

# insert after main capex graphs
figs.insert(2, fig)


print("Preparing load growth map")
# make a map of load growth by ReEDS region and insert after capex map
import json
import geopandas as gpd

# has columns WKT, rb (zone); filter to US zones only and convert to geometry and load_zone
# a shapefile with more info (state, RTO, interconnect and major region) is at
# https://github.com/NREL/ReEDS-2.0/tree/main/inputs/shapefiles/US_PCA
df = pd.read_csv(
    "https://raw.githubusercontent.com/NREL/ReEDS-2.0/refs/heads/main/inputs/shapefiles/US_CAN_MEX_PCA_polygons.csv"
)
df = df[df["rb"].str[1:].astype(int) <= 134]
zones = (
    gpd.GeoDataFrame(
        df,
        geometry=gpd.GeoSeries.from_wkt(df["WKT"]),
        crs="EPSG:4326",
    )
    .rename(columns={"rb": "load_zone"})
    .drop(columns="WKT")
    .set_index("load_zone", drop=False)
)

growth = pd.read_csv("../growth_rates/zone_growth.csv").set_index("load_zone")
for col, growth_col in [
    ("2025-35 avg load growth (%)", "avg_growth"),
    ("2025-35 peak load growth (%)", "peak_growth"),
]:
    zones[col] = zones["load_zone"].map(growth[growth_col])

    # Polygons
    fig = px.choropleth_map(
        zones,
        geojson=json.loads(zones.to_json()),
        locations=zones.index,  # unique id per feature
        color=col,
        # opacity=0.5,
        title=f"{col} by model zone",
    )
    fig.update_traces(marker_line_width=0.5, marker_line_color="white")

    # Center/zoom
    minx, miny, maxx, maxy = zones.total_bounds
    fig.update_layout(
        # map_style="open-street-map",  # no token needed
        map_zoom=3.3,  # exact level depends on browser width
        map_center={"lat": (miny + maxy) / 2, "lon": (minx + maxx) / 2},
        margin=dict(l=0, r=0, t=0, b=0),
    )

    # add to list of graphs
    figs.append(fig)


# %%
# put all the figures into one HTML file
html_parts = []
for i, fig in enumerate(figs):
    html = pio.to_html(
        fig,
        include_plotlyjs="cdn" if i == 0 else False,
        full_html=False,
        config={"scrollZoom": False},
    )
    # make everything a shorter
    html = f"""<div style="aspect-ratio: 2 / 1; width: 100%;">{html}</div>"""
    html_parts.append(html)

with open(out_html, "w") as f:
    f.write(
        (
            f"<html><head><title>Summary Figures - {label}</title></head><body>"
            + "\n".join(html_parts)
            + "</body></html>"
        )
    )
print(f"saved {out_html}")

# # make 100 MW clusters for easier population plotting
# reps = (info["capacity"] / 100).round().astype("Int64").fillna(1)
# # note: NA means no upper limit; we treat each zone as one data point
# info_dots = info.reindex(index=info.index.repeat(reps))

# fig = px.box(
#     info_dots,
#     x="group_wrapped",
#     y="gen_overnight_cost",
#     points=False,  # "outliers",  # or 'all', 'suspectedoutliers', or False
#     title="Overnight Cost by Technology and Build Year",
#     labels={
#         "group": "Group",
#         "gen_overnight_cost": "Overnight Cost ($/MW)",
#     },
# )
# fig.update_layout(xaxis_tickangle=90)
# fig.show(renderer="browser")

# %%
