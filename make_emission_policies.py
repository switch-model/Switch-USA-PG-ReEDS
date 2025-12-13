# %% #####################################
# setup
print("Loading libraries")
import sys, os, re, fnmatch
from pathlib import Path
from collections import defaultdict

import pandas as pd

try:
    # prevent warnings when importing geopandas if PyGEOS and shapely are both available
    import shapely

    os.environ["USE_PYGEOS"] = "0"
except:
    pass
import geopandas as gpd

from powergenome.generators import load_860m
from powergenome.util import load_settings

from utilities import read_yaml, write_yaml, add_yaml_key, delete_yaml_keys

print("Loaded libraries")

settings_path = Path("pg/settings")

settings = load_settings(settings_path)

region_info = gpd.read_file(
    Path(settings["input_folder"]) / settings["user_region_geodata_fn"]
)
# convert to capitals to match other ReEDS files
region_info["st"] = region_info["st"].str.upper()

# which yaml files contain each of the settings changed by this script
yaml_files = {
    "scenario_settings": settings_path / "scenario_management.yml",
    "model_tag_names": settings_path / "resource_tags.yml",
    "regional_tag_values": settings_path / "regional_resource_tags.yml",
    "generator_columns": settings_path / "model_definition.yml",
}

# map between PG tech search keys and ReEDS technologies
pg_reeds_tech_map = pd.read_csv(
    Path(settings["input_folder"]) / "pg_reeds_tech_map.csv"
)

# last year of historical data (e.g., in EIA 860m)
last_hist_year = 2024

max_growth_limits = [
    # tag, selector, limit, description
    # selector identifies existing plants of this type in the EIA 860M spreadsheet
    # limit can be a number or a tuple of number of historical years to consider
    # and an aggregation function to apply to annual totals
    # wind: use maximum over 10 years before last_hist_year
    # (14490 MW in 2020 in EIA 860M)
    (
        "MaxCapTag_WindGrowth",
        "energy_source_code_1 == 'WND'",
        # (10, "max"),
        14490,
        "National Wind Growth Limit",
    ),
    # solar: start with 2024 additions (30792.3) and allow 20%/year growth from
    # there forward (TODO: require construction in one period to enable growth
    # in the next)
    (
        "MaxCapTag_SolarGrowth",
        "technology_description == 'Solar Photovoltaic'",
        lambda y: 30792.3 * 1.2 ** (y - 2024),
        "National Solar Growth Limit",
    ),
    # Nuclear: no new build possible before 2035; up to 10 GW possible in 2035,
    # rising by 20%/year thereafter (based on general market assessment)
    (
        "MaxCapTag_NuclearGrowth",
        "technology_description == 'Nuclear'",
        lambda y: 0 if y < 2035 else 10000 * 1.2 ** (y - 2035),
        "National Nuclear Growth Limit",
    ),
    # Gas: 58 GW limit for 2025-2030 based on
    # https://www.woodmac.com/press-releases/coal-and-gas-generation-can-accommodate-40-to-75-of-expected-us-peak-demand-growth-through-20302/
    # and anecdotal reports that gas turbine supply chains are maxed out in the 10-20 GW/year range
    # see, e.g.,
    # https://pages.marketintelligence.spglobal.com/rs/565-BDO-100/images/Global%20gas%20turbine%20manufacturing%20faces%20soaring.pdf
    # https://www.utilitydive.com/news/mitsubishi-gas-turbine-manufacturing-capacity-expansion-supply-demand/759371/
    # https://gridlab.org/portfolio-item/gas-tubine-cost-report/
    # https://rmi.org/gas-turbine-supply-constraints-threaten-grid-reliability-more-affordable-near-term-solutions-can-help/
    (
        "MaxCapTag_GasTurbineSupply",
        "technology_description.isin(['Natural Gas Fired Combined Cycle', 'Natural Gas Fired Combustion Turbine'])",
        58000 / (2030 - 2024),
        "Gas Turbine Supply-Chain Limit",
    ),
]

# keep files small and uncluttered
possible_model_years = list(range(2024, 2030)) + list(range(2030, 2050 + 1, 5))

# useful for sorting dataframes by region number instead of alphabetically
region_sort = dict(by="region", key=lambda r: r.str[1:].astype(int))


def update_model_tag_names(new_tags, prefix=None):
    """
    delete all tags from model_tag_names and generator_columns lists if they
    match the prefix, then add all new_tags to the prefix. save the file if
    updated.
    """
    for key in ["model_tag_names", "generator_columns"]:
        ym = read_yaml(yaml_files[key])
        tags = ym[key]
        old = set(tags)
        # delete matching tags in place, starting at the end and moving up
        # (to preserve order and any comments)
        if isinstance(prefix, str):
            for i in range(len(tags) - 1, -1, -1):
                if tags[i].startswith(prefix):
                    del tags[i]
        # add new tags
        tags.extend(sorted(set(new_tags)))
        if old != set(tags):
            print(f"updated {key}.")
            write_yaml(ym, yaml_files[key])


# %% #####################################
# ESR target
# columns: case_id,year,region,ESR_1,ESR_2,UREC_Limit_ESR_1,UREC_Limit_ESR_2
# note: UREC_Limit columns are an extension for Switch and will probably be
# ignored when creating GenX inputs

"""
create extra_inputs/current_emission_policies.csv
with current state and regional level clean energy
standards and carbon limits

output file has structure 
case_id,year,region,ESR_1,ESR_2,CO_2_Cap_Zone_1,CO_2_Max_Mtons_1,CO_2_Max_tons_MWh_1,UREC_Limit_ESR_1,UREC_Limit_ESR_2
all,2030,NENGREST,0.529,0.289,1,8.59,-1
all,2030,NENG_CT,0.634,0,1,2.31,-1
all,2030,NENG_ME,0.819,0,1,1.29,-1
"""

esr_dfs = []
for prog in ["ces", "rps"]:
    # prog = 'rps'

    frac = pd.read_csv(
        f"https://raw.githubusercontent.com/NREL/ReEDS-2.0/refs/heads/main/inputs/state_policies/{prog}_fraction.csv"
    )
    # do various possible renames
    frac = frac.rename(
        columns={"*t": "year", "t": "year", "rps_all": "rps", "Value": "ces"}
    )
    # keep only the years we're interested in
    frac = frac.query("year.isin(@possible_model_years)")
    # convert to separate rows for each program (year, st, prog (rps, ces, etc), target)
    frac = frac.melt(id_vars=["year", "st"], var_name="prog", value_name="target")
    frac["program"] = "ESR_" + frac["st"] + "_" + frac["prog"]
    # drop empty/inactive programs
    frac = frac.query("target > 0")
    esr_dfs.append(frac)

    # add limits on out-of-state fractions for URECs as if they were a different
    # ESR (starting with "UREC_Limit_")
    oos_limit = pd.read_csv(
        "https://raw.githubusercontent.com/NREL/ReEDS-2.0/refs/heads/main/inputs/state_policies/oosfrac.csv"
    ).set_index("*st")["value"]
    oos = (
        frac.assign(program="UREC_Limit_" + frac["program"])
        .assign(target=frac["st"].map(oos_limit))
        .dropna(subset="target")
    )
    esr_dfs.append(oos)


esr_long = pd.concat(esr_dfs).merge(region_info[["st", "region"]], on="st")
esr_wide = esr_long.pivot(index=["year", "region"], values="target", columns="program")
esr_wide = esr_wide[sorted(esr_wide.columns)]

# %% #####################################
# ESR eligibility
# we follow ReEDS's approach that everything labeled RE in
# https://github.com/NREL/ReEDS-2.0/blob/main/inputs/tech-subset-table.csv
# is RPS-eligible unless listed in
# https://github.com/NREL/ReEDS-2.0/blob/main/inputs/state_policies/techs_banned_rps.csv
# Then, within each state, everything that is RPS-eligible is also CES eligible,
# plus everything labeled NUCLEAR, HYDRO, CCS or CANADA, except those listed in
# https://github.com/NREL/ReEDS-2.0/blob/main/inputs/state_policies/techs_banned_ces.csv
# (see https://github.com/NREL/ReEDS-2.0/blob/92e8fa7cc9f870006ca2df52d98fd11f1db68dbe/b_inputs.gms#L3087)
# After gathering eligible ReEDS technologies, we convert them to PG technology terms
# to generate the eligibility flags (as a .yml file)
# TODO: finer-scale CES rules later in the file, based on emission rates of technologies
techs = pd.read_csv(
    "https://github.com/NREL/ReEDS-2.0/raw/refs/heads/main/inputs/tech-subset-table.csv"
)
techs = techs.rename(columns={techs.columns[0]: "reeds_tech"})

# fmt: off
# convert tech ranges like dr_shed_1*dr_shed_2 into separate rows
pat = re.compile(r"^(?P<prefix>.+?)_(?P<start>\d+)\*(?P=prefix)_(?P<stop>\d+)$")
def expand_label(s: str) -> list[str]:
    """
    convert "tech_x_1*tech_x_3" into ["tech_x_1", "tech_x_2", "tech_x_3"]
    """
    m = pat.match(s)
    if not m:
        return [s]  # no range; keep as-is
    pre = m.group("prefix")
    start = int(m.group("start"))
    stop = int(m.group("stop"))
    return [f"{pre}_{i}" for i in range(start, stop + 1)]
# fmt: on


techs["reeds_tech_list"] = techs["reeds_tech"].apply(expand_label)
techs = techs.explode("reeds_tech_list", ignore_index=True)
techs["reeds_tech"] = techs.pop("reeds_tech_list")

re_techs = techs.query("RE == 'YES'")["reeds_tech"]
solar_techs = techs.query("PV == 'YES' or PVB == 'YES'")["reeds_tech"]
wind_techs = techs.query("WIND == 'YES'")["reeds_tech"]
ce_techs = techs.loc[
    techs[["RE", "NUCLEAR", "HYDRO", "CCS", "CANADA"]].notna().any(axis=1), "reeds_tech"
]
techs_banned_ces = pd.read_csv(
    "https://github.com/NREL/ReEDS-2.0/raw/refs/heads/main/inputs/state_policies/techs_banned_ces.csv"
)
techs_banned_rps = pd.read_csv(
    "https://github.com/NREL/ReEDS-2.0/raw/refs/heads/main/inputs/state_policies/techs_banned_rps.csv"
)
# virtual ban on can-imports in states bordering Mexico, because Canadian and
# Mexican imports will both be interpreted as generic imports but imports from
# Mexico are ineligible for RPS
techs_banned_rps.loc[
    techs_banned_rps["i"] == "can-imports", ["CA", "AZ", "NM", "TX"]
] = 1

rps_ban = {
    st: techs_banned_rps.loc[col.notna(), "i"]
    for st, col in techs_banned_rps.iloc[:, 1:].items()
}
ces_ban = {
    st: techs_banned_ces.loc[col.notna(), "i"]
    for st, col in techs_banned_ces.iloc[:, 1:].items()
}
# eligible techs for each type of program; note that the rps bans apply to both
# rps and ces techs
prog_rules = {
    # prog: (starting techs, bans to apply)
    "rps": (re_techs, [rps_ban]),
    "rps_solar": (solar_techs, [rps_ban]),
    "rps_wind": (wind_techs, [rps_ban]),
    "ces": (ce_techs, [rps_ban, ces_ban]),
}

# Build dicts showing which states can send RECs to each state on a bundled or unbundled basis
# allowed trade between states (0=none, 1=bundled+unbundled, 2=bundled)
# trade direction is col (ast) -> row (st)
# see https://github.com/NREL/ReEDS-2.0/blob/92e8fa7cc9f870006ca2df52d98fd11f1db68dbe/b_inputs.gms#L3034)
# note: the table allows trade from each state to itself, but we treat that as local, so ignore
trade_table = pd.read_csv(
    "https://raw.githubusercontent.com/NREL/ReEDS-2.0/refs/heads/main/inputs/state_policies/rectable.csv",
)
trade_partners = trade_table.melt(
    id_vars="st", var_name="ast", value_name="trade"
).query("st != ast and trade > 0")
bundled_partners = defaultdict(list)
unbundled_partners = defaultdict(list)
for r in trade_partners.itertuples():
    bundled_partners[r.st].append(r.ast)
    if r.trade == 1:
        unbundled_partners[r.st].append(r.ast)

# build dataframes with eligible techs for each state/program, incl. REC trade
el_dfs = []
for st, grp in esr_long[["st", "prog"]].drop_duplicates().groupby("st"):
    for idx, st, prog in grp.itertuples():
        techs, bans = prog_rules[prog]
        # filter out unwanted techs for this st
        for ban in bans:
            techs = techs[~techs.isin(ban.get(st, []))]
        local_techs = pd.DataFrame(
            {"st": st, "program": f"ESR_{st}_{prog}", "reeds_tech": techs}
        )
        el_dfs.append(local_techs)
        # add trade partners (create rows indicating that the same techs in a partner state
        # can ship recs to a version of this program with "_bundled" or "_unbundled" appended,
        # e.g., techs that would be eligible for ESR_CA_rps but are in AZ are
        # marked eligible for ESR_CA_rps_bundled and ESR_CA_rps_unbundled)
        # note: "X_bundled" and "X_unbundled" columns are an extension for Switch;
        # they will probably be automatically omitted from GenX outputs since they
        # don't have a corresponding requirement target (and trade will not be possible)
        for tag, partners in [
            ("bundled", bundled_partners[st]),
            ("unbundled", unbundled_partners[st]),
        ]:
            for ast in partners:
                techs = local_techs.copy()
                techs["st"] = ast
                techs["program"] += "_" + tag
                el_dfs.append(techs)

esr_el = pd.concat(el_dfs, ignore_index=True)

esr_el = esr_el.merge(pg_reeds_tech_map, on="reeds_tech")
esr_el = esr_el.merge(region_info[["st", "region"]], on="st")
esr_el["region_num"] = esr_el["region"].str[1:].astype(int)

# map to PowerGenome techs and ReEDS regions
# create eligibility tree for prog:
# regional_tag_values:
#   p1:
#     ESR_CT_rps:
#       UtilityPV: 1
#       LandbasedWind: 1

# open regional_resource_tags file, remove any existing ESR tags,
# then add tags eligible gens for every region in every program
rrt = read_yaml(yaml_files["regional_tag_values"])
delete_yaml_keys(rrt, ["regional_tag_values", "*", "ESR_*"])
for (rno, r), g1 in esr_el.groupby(["region_num", "region"]):
    for p, g2 in g1.groupby("program"):
        d = {t: 1 for t in g2["pg_tech"]}
        add_yaml_key(rrt, ["regional_tag_values", r, p], d)
write_yaml(rrt, yaml_files["regional_tag_values"])

# add the tags to model_tag_names
update_model_tag_names(esr_el["program"], "ESR_")

# %% #####################################
# carbon targets: case_id,year,region,CO_2_Cap_Zone_1,CO_2_Max_Mtons_1,CO_2_Cap_Zone_2,CO_2_Max_Mtons_2,...
# see policies here:
# https://icapcarbonaction.com/en/ets
# https://github.com/NREL/ReEDS-2.0/tree/main/inputs/emission_constraints
# https://github.com/NREL/ReEDS-2.0/blob/main/inputs/emission_constraints/rggi_states.csv
# https://github.com/NREL/ReEDS-2.0/blob/main/inputs/emission_constraints/rggicon.csv

# cap: RGGI
# taxes: CA, WA, NY (not in effect yet, but we act like it is)
# Future:
# PA is supposed to enter RGGI eventually, but not clear when or with what target

tax_states = ["CA", "WA", "NY"]

# start with the RGGI caps, then add CA and WA with a zero cap (they'll just pay
# the slack price)
rggi_cap = pd.read_csv(
    "https://github.com/NREL/ReEDS-2.0/raw/refs/heads/main/inputs/emission_constraints/rggicon.csv",
    names=["year", "cap"],
    header=None,
).query("year.isin(@possible_model_years)")
# convert from tonnes to million tonnes
rggi_cap["cap"] *= 0.000001
rggi_states = pd.read_csv(
    "https://github.com/NREL/ReEDS-2.0/raw/refs/heads/main/inputs/emission_constraints/rggi_states.csv"
).rename(columns={"*st": "st"})

co2 = (
    rggi_cap.assign(dummy=1)
    .merge(rggi_states.assign(dummy=1), on="dummy")
    .drop(columns="dummy")
)
# tag these states as being in CO2 program 1
# note: same cap will be shown for every state/region, but we zero out
# all but one region later
co2["CO_2_Cap_Zone_1"] = 1
co2 = co2.rename(columns={"cap": "CO_2_Max_Mtons_1"})
co2_long = co2.melt(id_vars=["year", "st"], var_name="col", value_name="value")

# now add other programs
tax_years = co2_long["year"].unique()
# 1 = in program
tag_func = lambda i: (f"CO_2_Cap_Zone_{i}", 1)
# 0 = pay tax on all tonnes (0 free allocation)
target_func = lambda i: (f"CO_2_Max_Mtons_{i}", 0)
taxes = pd.DataFrame(
    [
        (y, st) + f(i + 2)  # like (2025, 'CA', 'CO_2_Cap_Zone_2', 0)
        for (i, st) in enumerate(tax_states)
        for y in tax_years
        for f in [tag_func, target_func]
    ],
    columns=["year", "st", "col", "value"],
)

co2_long = pd.concat([co2_long, taxes])

co2_reg = co2_long.merge(region_info[["st", "region"]], on="st").drop(columns="st")
co2_wide = co2_reg.pivot(columns="col", index=["year", "region"], values="value")
co2_cols = sorted(co2_wide.columns.to_list(), key=lambda c: (int(c.split("_")[-1]), c))
co2_wide = co2_wide[co2_cols]

# create alternative co2 frame with deep decarbonization everywhere
decarb_co2_wide = pd.concat(
    [
        pd.DataFrame(
            {
                "year": y,
                "region": region_info["region"],
                "CO_2_Cap_Zone_1": 1,
                "CO_2_Max_Mtons_1": 0,
            }
        )
        for y in possible_model_years
    ]
).set_index(["year", "region"])

# %% #####################################
# Offshore wind mandates (target level and generator eligibility)

# These are both added to the settings as yaml keys. Note that the PG MinCapReq
# key doesn't include year indexing, so it has to be changed year by year by
# defining it under the settings_management key:
# settings_management:
#   2030:
#     all_cases:
#       MinCapTag_1:
#         description: NY_offshorewind
#         min_mw: 1054
#   2035:
#     all_cases:
#       MinCapTag_1:
#         description: NY_offshorewind
#         min_mw: 1054
# This requires interleaving the new settings with whatever is there already, so
# we use ruamel.yaml to write the new data, preserving existing comments and
# quotes.

# Eligibility is written similarly to ESR eligibility (and is interleaved with it):
# regional_tag_values:
#   p10:
#     MinCapTag_CA_offshorewind:
#       OffShoreWind: 1
#       offshore wind: 1

# "st", "2019", "2020", ...
# CA, 0, 0, 100, ...
osw_req = pd.read_csv(
    "https://github.com/NREL/ReEDS-2.0/raw/refs/heads/main/inputs/state_policies/offshore_req_default.csv"
)
osw_req["program"] = "MinCapTag_" + osw_req["st"] + "_offshorewind"

# update YAML 'settings_management' key, preserving existing quotes and comments
ss = read_yaml(yaml_files["scenario_settings"])

# remove/report any previous all_cases MinCapReq entries
delete_yaml_keys(ss, ["settings_management", "*", "all_cases", "MinCapReq"])

# add new all_cases MinCapReq entries as needed
for y in possible_model_years:
    d = {
        program: {"description": f"{st}_offshorewind", "min_mw": target}
        for i, st, program, target in osw_req[["st", "program", str(y)]].itertuples()
        if target > 0
    }
    add_yaml_key(ss, ["settings_management", y, "all_cases", "MinCapReq"], d)

write_yaml(ss, yaml_files["scenario_settings"])

# write eligibility (anything in reeds_ tech_map matching ReEDS wind-ofs_1)
rrt = read_yaml(yaml_files["regional_tag_values"])

# remove existing eligibility info for any MinCapTag programs
delete_yaml_keys(rrt, ["regional_tag_values", "*", "MinCapTag_*"])

# eligibility tags like {'OffShoreWind': 1, 'offshore wind': 1} (same in every region)
osw_el_dict = {
    t: 1 for t in pg_reeds_tech_map.query("reeds_tech == 'wind-ofs_1'")["pg_tech"]
}

# programs and regions they include (e.g., MinCapTag_CA_offshorewind, p10)
osw_region = osw_req.merge(region_info)[["program", "region"]]
osw_region = osw_region.sort_values(**region_sort)

# store eligibility data
for i, program, region in osw_region.itertuples():
    add_yaml_key(rrt, ["regional_tag_values", region, program], osw_el_dict)

write_yaml(rrt, yaml_files["regional_tag_values"])

# add the tags to model_tag_names
update_model_tag_names(osw_region["program"], "MinCapTag_")

# %% #####################################
# merge ESRs and CO2 targets and save to input_folder
for file, cdf in [
    ("emission_policies_current.csv", co2_wide),
    ("emission_policies_decarb.csv", decarb_co2_wide),
]:
    ep = pd.concat([esr_wide, cdf], axis=1)

    # fix up the row ordering and some column types
    ep = ep.loc[sorted(ep.index.to_list(), key=lambda r: (r[0], int(r[1][1:]))), :]
    for c in ep.columns:
        if c.startswith("CO_2_Cap_Zone_"):
            ep[c] = ep[c].astype("Int64")

    # zero out the targets for all but the first region in each co2 program each year
    for c in ep.columns:
        if c.startswith("CO_2_Max_Mtons_"):
            nz = ep[c] > 0
            pos_in_year = nz.groupby(level="year").cumsum()
            ep.loc[nz & (pos_in_year > 1), c] = 0

    # use some magic to add a case_id="all" level at the start of the index
    # (wraps the existing dataframe under a new outer level 'all')
    ep = pd.concat({"all": ep}, names=["case_id"])

    # fill blanks because PG drops rows with any NA; 0=not in cap or no ESR goal
    ep = ep.fillna(0)

    ep_file = Path(settings["input_folder"]) / file
    ep.to_csv(ep_file, index=True)
    print(f"Saved {ep_file}.")

# %% #####################################
# Apply maximum growth rate for wind and gas (net of planned growth in EIA 860m)
# as MaxCapReq values for each possible model year like
# settings_management:
#   2025:
#     all_cases:
#       MaxCapReq:
#         MaxCapTag_WindGrowth:
#           description: National Wind Growth Limit
#           max_mw: 13000
#         MaxCapTag_GasTurbineSupply:
#           description: Gas Turbine Supply-Chain Limit
#           max_mw: 9700

# get relevant settings and clear any existing tags of this type
ss = read_yaml(yaml_files["scenario_settings"])
delete_yaml_keys(ss, ["settings_management", "*", "all_cases", "MaxCapReq"])


def make_const_func(lim):
    return lambda y: lim


wb = load_860m(settings)
lims = []
for tag, selector, limit, description in max_growth_limits:
    subset = wb["operating"].query(selector)
    if isinstance(limit, tuple):
        # apply limit query
        n_years, agg = limit
        annual_limit = float(
            subset.query(f"operating_year >= {last_hist_year - n_years + 1}")
            .groupby("operating_year")["winter_capacity_mw"]
            .sum()
            .agg(agg)
        )
        lim_func = make_const_func(annual_limit)
    elif isinstance(limit, (int, float)):
        lim_func = make_const_func(limit)
    elif callable(limit):
        lim_func = limit
    else:
        raise ValueError(f"Unknown type for limit {limit}: {type(limit)}")
    # TODO: maybe have different baselines for each future year as retirements roll through?
    baseline_capacity = float(
        subset.query(
            f"operating_year <= {last_hist_year} and planned_retirement_year.isna()"
        )["winter_capacity_mw"].sum()
    )
    lims.append((tag, description, lim_func, baseline_capacity))

# write caps to the settings_management yaml entry in format above
for y in possible_model_years:
    d = {}
    for tag, description, lim_func, baseline_capacity in lims:
        # cap is whatever already exists plus max possible growth to this year
        max_capacity = baseline_capacity + sum(
            lim_func(y_ + 1) for y_ in range(last_hist_year, y)
        )
        d[tag] = {
            "description": description,
            "max_mw": max_capacity,
        }
        update_model_tag_names([tag], tag)
    add_yaml_key(ss, ["settings_management", y, "all_cases", "MaxCapReq"], d)

write_yaml(ss, yaml_files["scenario_settings"])

# %%
