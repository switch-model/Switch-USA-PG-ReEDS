# %% #####################################
# setup

"""
create extra_inputs/current_emission_policies.csv
with current state and regional level clean energy
standards and carbon limits

output file has structure
case_id,year,region,ESR_1,ESR_2,CO_2_Cap_Zone_1,CO_2_Max_Mtons_1,CO_2_Max_tons_MWh_1
all,2030,NENGREST,0.529,0.289,1,8.59,-1
all,2030,NENG_CT,0.634,0,1,2.31,-1
all,2030,NENG_ME,0.819,0,1,1.29,-1
"""

print("Loading libraries")
import sys, os, re, fnmatch
from pathlib import Path

import pandas as pd
import ruamel.yaml
from ruamel.yaml.comments import CommentedMap, CommentedSeq

from powergenome.generators import load_860m

try:
    # prevent warnings when importing geopandas if PyGEOS and shapely are both available
    import shapely

    os.environ["USE_PYGEOS"] = "0"
except:
    pass
import geopandas as gpd

from powergenome.util import load_settings

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

# tag to use to limit total wind in the U.S. based on historical growth rates
max_wind_growth_tag = "MaxCapTag_WindGrowth"


# keep files small and uncluttered
possible_model_years = list(range(2024, 2030)) + list(range(2030, 2050 + 1, 5))

# setup yaml parser to preserve quotes and disable aliasing (cross-referencing)
# of duplicate data; also preserves comments by default.
ym = ruamel.yaml.YAML()
ym.preserve_quotes = True
ym.representer.ignore_aliases = lambda x: True

# useful for sorting dataframes by region number instead of alphabetically
region_sort = dict(by="region", key=lambda r: r.str[1:].astype(int))


def read_yaml(file, quiet=False):
    with open(file, "r") as f:
        yaml_obj = ym.load(f)
    if not quiet:
        print(f"read {file}.")
    return yaml_obj


def write_yaml(yaml_obj, file, quiet=False):
    with open(file, "w") as f:
        ym.dump(yaml_obj, f)
    if not quiet:
        print(f"updated {file}.")


def delete_yaml_keys(yaml_root, path, quiet=False):
    """
    Delete keys in a ruamel.yaml mapping following a path that may include '*'
    (wildcard for 'all keys' at that level). For each matched endpoint:
      - remove the value AND any comment attached to that key
      - print a message like "deleted a:b:c" and append ": *" if the deleted
        value was a container (mapping/sequence).

    Parameters
    ----------
    yaml_root : CommentedMap | dict
        The ruamel.yaml (comment-preserving) root mapping.
    path : list
        A list of path segments, e.g. ['settings_management', '*', 'all_cases', 'MinCapReq'].
        Use '*' to mean "all keys" at that level.
    quiet : bool
        If False, print a deletion line for each removed key.
    """

    def _recurse(obj, path, trail, parent=None):
        # object to recurse into, path to look for, trail we followed to get here
        if not path:
            # end of the line, delete last element of trail from obj
            key = trail[-1]
            try:
                # remove comment attached to this key (if any)
                parent.ca.items.pop(key, None)
            except AttributeError:  # rare
                pass
            # remove the entry itself
            val = parent.pop(key)
            if not quiet:
                msg = f"deleted {': '.join(str(p) for p in trail)}"
                if isinstance(val, (CommentedMap, CommentedSeq, dict, list)):
                    msg += ": *"
                print(msg)
            return

        head, *tail = path

        if isinstance(obj, (CommentedMap, dict)):
            # iterate over a snapshot of matching keys (we may delete under children)
            for k in list(obj.keys()):
                if fnmatch.fnmatch(str(k), str(head)):
                    # match on name or wildcard (*, ?)
                    _recurse(obj[k], tail, trail + [k], obj)

        elif isinstance(obj, (CommentedSeq, list, tuple)):
            # iterate over the specified slice, either "*" for all indices
            # or a single int or a slice object (probably going overboard here)
            if isinstance(head, int):
                s = slice(head, None)
            elif head == "*":
                s = slice()
            elif isinstance(head, slice):
                s = head
            else:
                # could add code here to handle special cases like "3:5"
                raise ValueError(f"Unexpected index for sequence: {head}")

            # process selected elements of obj
            for idx in range(*s.indices(len(obj))):
                _recurse(obj[idx], tail, trail + [idx], obj)

    _recurse(yaml_root, list(path), [])


def add_yaml_key(yaml_root, path, value, quiet=False):
    """
    Add the specified value to a ruamel.yaml mapping at the specified path,
    creating nodes needed to along the way. Also add a comment to the key,
    "managed by make_emission_policies.py"

    Parameters
    ----------
    yaml_root : CommentedMap | dict
        The ruamel.yaml (comment-preserving) root mapping.
    path : list
        A list of path segments, e.g. ['settings_management', 2025, 'all_cases', 'MinCapReq'].
    value : object
       The value to be stored at that location, possibly a dict or sequence.
    quiet : bool
        If False, print a message for each added key.
    """
    obj = yaml_root
    # find the parent node, creating any needed along the way
    for node in path[:-1]:
        obj = obj.setdefault(node, CommentedMap())

    obj[path[-1]] = value
    obj.yaml_add_eol_comment("managed by make_emission_policies.py", path[-1])
    if not quiet:
        msg = f"added {': '.join(str(p) for p in path)}"
        if isinstance(value, (CommentedMap, CommentedSeq, dict, list)):
            msg += ": *"
        print(msg)


def update_model_tag_names(new_tags, prefix=None):
    # delete all tags from model_tag_names and generator_columns lists if
    # they match the prefix, then add all new_tags to the prefix. save the file
    # if updated.
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
# columns: case_id,year,region,ESR_1,ESR_2

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
# TODO: finer-scale CES rules later in the file, based on emission rates of technologies
# After gathering eligible ReEDS technologies, we convert them to PG technology terms
# to generate the eligibility flags (as a .yml file)
techs = pd.read_csv(
    "https://github.com/NREL/ReEDS-2.0/raw/refs/heads/main/inputs/tech-subset-table.csv"
)
techs = techs.rename(columns={techs.columns[0]: "reeds_tech"})

# convert tech ranges like dr_shed_1*dr_shed_2 into separate rows
pat = re.compile(r"^(?P<prefix>.+?)_(?P<start>\d+)\*(?P=prefix)_(?P<stop>\d+)$")


def expand_label(s: str) -> list[str]:
    # convert "tech_x_1*tech_x_3" into ["tech_x_1", "tech_x_2", "tech_x_3"]
    m = pat.match(s)
    if not m:
        return [s]  # no range; keep as-is
    pre = m.group("prefix")
    start = int(m.group("start"))
    stop = int(m.group("stop"))
    return [f"{pre}_{i}" for i in range(start, stop + 1)]


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

el_dfs = []
for st, grp in esr_long[["st", "prog"]].drop_duplicates().groupby("st"):
    # make dataframes with eligible techs for each state/program
    for idx, st, prog in grp.itertuples():
        techs, bans = prog_rules[prog]
        # filter out unwanted techs for this st
        for ban in bans:
            techs = techs[~techs.isin(ban.get(st, []))]
        el_dfs.append(
            pd.DataFrame({"st": st, "program": f"ESR_{st}_{prog}", "reeds_tech": techs})
        )
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
# calculate maximum historical growth rate for wind, then store that
# as a MaxCapReq for each possible model year like
# settings_management:
#   2025:
#     all_cases:
#       MaxCapReq:
#         MaxCapTag_WindGrowth:
#           description: National Wind Growth Limit
#           max_mw: 14000

wb = load_860m(settings)
wind = wb["operating"].query('energy_source_code_1 == "WND" and operating_year >= 2015')
max_wind_growth = float(
    wind.groupby("operating_year")["winter_capacity_mw"].sum().max()
)
baseline_wind = float(wind.query("operating_year <= 2024")["winter_capacity_mw"].sum())

# write caps to the settings_management yaml entry in format above
ss = read_yaml(yaml_files["scenario_settings"])
delete_yaml_keys(ss, ["settings_management", "*", "all_cases", "MaxCapReq"])
for y in possible_model_years:
    # cap is whatever already exists plus max possible growth to this year
    max_wind = baseline_wind + (y - 2024) * max_wind_growth
    d = {
        max_wind_growth_tag: {
            "description": "National Wind Growth Limit",
            "max_mw": max_wind,
        }
    }
    add_yaml_key(ss, ["settings_management", y, "all_cases", "MaxCapReq"], d)
write_yaml(ss, yaml_files["scenario_settings"])
update_model_tag_names([max_wind_growth_tag], max_wind_growth_tag)


# %%
