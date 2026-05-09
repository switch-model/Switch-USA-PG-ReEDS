print("Loading libraries.")
import re, os
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.optimize import linprog
from ruamel.yaml.comments import CommentedSeq

from powergenome.generators import load_860m
from powergenome.util import load_settings
from powergenome.params import DATA_PATHS

from utilities import read_yaml, write_yaml, add_yaml_key, delete_yaml_keys

print("Finished loading libraries.")

settings_dir = "pg/settings"
addl_retirements_file = Path(settings_dir) / "resources.yml"


settings = load_settings(settings_dir)
eia_860m_fn = settings.get("eia_860m_fn")
if not eia_860m_fn:
    raise ValueError(f"No `eia_860m_fn` parameter defined in {settings_dir}")

gem_fn = settings.get("GEM_coal_tracker_fn")
if not gem_fn:
    raise ValueError(f"No `GEM_coal_tracker_fn` parameter defined in {settings_dir}")

# GEM_coal_tracker_url = settings.get("GEM_coal_tracker_url")
GEM_coal_tracker_file = Path(settings["input_folder"]) / gem_fn

abs_860 = DATA_PATHS["eia_860m"]
path_860 = abs_860
if_shorter = lambda p: p if len(str(p)) < len(str(path_860)) else path_860
# shorten the path name if we can
try:
    path_860 = if_shorter(abs_860.relative_to(Path.cwd()))
except:
    pass
if os.name == "posix":
    try:
        path_860 = if_shorter("~" / abs_860.relative_to(Path.home()))
    except:
        pass

print(f"Reading 860M workbook from {path_860 / eia_860m_fn}.")

# Use 860M workbook specified for PowerGenome, but rename some columns
# back to the EIA version for easier cross-referencing to the workbook
data_dict = load_860m(settings)
eia = (
    data_dict["operating"]
    .rename(
        columns={
            "plant_id_eia": "Plant ID",
            "plant_name": "Plant Name",
            "generator_id": "Generator ID",
            "capacity_mw": "Nameplate Capacity (MW)",
            "technology_description": "Technology",
            "operating_year": "Operating Year",
            "planned_retirement_year": "Planned Retirement Year",
            "latitude": "Latitude",
            "longitude": "Longitude",
        }
    )
    .copy()
)

# tag each generator with the corresponding row in the Excel sheet
# (4-based to match the view in Excel)
eia["eia_row"] = eia.index + 4  # 0->4
# only keep coal plants
eia = eia[eia["Technology"].str.contains("Coal")]
# GEM only covers plants >= 30 MW
eia = eia[eia["Nameplate Capacity (MW)"] >= 30]

# get data from GEM coal closures workbook
print(f"Reading GEM workbook from {GEM_coal_tracker_file}")
gem = pd.read_excel(GEM_coal_tracker_file, sheet_name="Units")
gem["gem_row"] = gem.index + 2  # 0->1 and skip header
gem = gem[gem["Country/Area"] == "United States"].copy()
# strip out extra "timepoint 1" notations
gem["Unit name"] = gem["Unit name"].str.replace(", timepoint 1", "")
# treat actual retirements as planned retirements (some of these are not in EIA)
gem["Planned retirement"] = gem["Planned retirement"].fillna(gem["Retired year"])

# Calculate the "distance" from each generator in 860m to each generator in
# GEM, based on
# candidates:
# - spatial distance < 3 miles (boolean)
# - size match < 1 (boolean)
# score:
# - first word of plant name matches (boolean)
# - unit digits match (boolean)
# - unit digits blank (boolean)
# - size difference
# - start year difference


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088  # mean Earth radius [km]
    φ1, λ1, φ2, λ2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    dφ = φ2 - φ1
    dλ = λ2 - λ1
    a = np.sin(dφ / 2) ** 2 + np.cos(φ1) * np.cos(φ2) * np.sin(dλ / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


print("Matching GEM generators to EIA generators.")

# calculate "distance" from each eia row to each gem row
distances = []
for i, e in eia.iterrows():
    # Only consider gem rows within 3 km
    g = gem[
        haversine_km(e["Latitude"], e["Longitude"], gem["Latitude"], gem["Longitude"])
        < 3
    ]
    # only consider plants within 25% of same size
    g = g[(g["Capacity (MW)"] / e["Nameplate Capacity (MW)"] - 1).abs() <= 0.25]
    # calculate weighted "distance" score
    d = 0
    # first word of plant name differs
    d = 10 * d + (g["Plant name"].str.split().str[0] != e["Plant Name"].split()[0])
    # unit digits differ
    d = 10 * d + (
        g["Unit name"].str.replace(r"\D+", "", regex=True)
        != re.sub(r"\D+", "", e["Generator ID"])
    )
    # size difference (MW)
    d = 1000 * d + (g["Capacity (MW)"] - e["Nameplate Capacity (MW)"]).abs()
    # start year difference (1 year treated as equivalent to 1 MW); sometimes missing for cancelled projects
    d = d + (g["Start year"] - e["Operating Year"]).abs().fillna(100)
    distances.extend((e["eia_row"], r_, d_) for d_, r_ in zip(d, g["gem_row"]))

# Use a linear program to assign eia rows to gem rows, minimizing a distance
# score. This prioritizes assigning each row to a partner, and then secondarily
# minimizing the sum of the distances between the partners.
penalty = 100000  # penalty for leaving rows unmatched
eia_rows = sorted({i for i, _, _ in distances})
gem_rows = sorted({j for _, j, _ in distances})

# Variables: [w_ij for all listed pairs] + [s_i for each eia id] + [t_j for each gem id]
pairs = [(i, j) for i, j, _ in distances]
nW, nE, nG = len(pairs), len(eia_rows), len(gem_rows)

# Objective: minimize sum w_ij * dist_ij + penalty * (sum s_i + sum t_j)
c_w = np.array([d for _, _, d in distances], dtype=float)
c_s = np.full(nE, float(penalty))
c_t = np.full(nG, float(penalty))
c = np.concatenate([c_w, c_s, c_t])

# Equality constraints:
# For each eia i: sum_j w_ij + s_i = 1
# For each gem j: sum_i w_ij + t_j = 1
A_eq = np.zeros((nE + nG, nW + nE + nG), dtype=float)
b_eq = np.ones(nE + nG, dtype=float)

eia_pos = {i: k for k, i in enumerate(eia_rows)}
gem_pos = {j: k for k, j in enumerate(gem_rows)}

# Rows for eia constraints
for k, (i, j) in enumerate(pairs):
    A_eq[eia_pos[i], k] += 1.0
# Add s_i coefficients (identity)
for i, r in eia_pos.items():
    A_eq[r, nW + r] = 1.0

# Rows for gem constraints
base = nE
for k, (i, j) in enumerate(pairs):
    A_eq[base + gem_pos[j], k] += 1.0
# Add t_j coefficients (identity)
for j, r in gem_pos.items():
    A_eq[base + r, nW + nE + r] = 1.0

# Bounds:
# 0 <= w_ij <= 1; 0 <= s_i <= 1; 0 <= t_j <= 1
bounds = [(0.0, 1.0)] * nW + [(0.0, 1.0)] * nE + [(0.0, 1.0)] * nG

res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
if not res.success:
    raise RuntimeError(f"LP failed: {res.message}")

x = res.x
w = x[:nW]
s = x[nW : nW + nE]
t = x[nW + nE :]

weights = {pair: w[idx] for idx, pair in enumerate(pairs)}
eia_shortfall = {i: s[eia_pos[i]] for i in eia_rows if s[eia_pos[i]] > 0}
gem_shortfall = {j: t[gem_pos[j]] for j in gem_rows if t[gem_pos[j]] > 0}

# make sure we got exact 1-1 matching
assert all(w in {0, 1} for w in weights.values())

# make dicts showing the winning matches
gem_for_eia = {e: g for (e, g), w in weights.items() if w == 1}
eia_for_gem = {g: e for (e, g), w in weights.items() if w == 1}

# report any that didn't match (for EIA it's one oddball one; for GEM, they
# all seem to be cases where the EIA plant is now marked as natural gas, not coal)
missing_candidates = set(eia["eia_row"]) - set(eia_rows)
if eia_shortfall or missing_candidates:
    print(f"Unmatched EIA rows:")
    for (e, g), d in zip(pairs, c_w):
        if e in eia_shortfall:
            alt_match = (
                f" (matches EIA row {eia_for_gem[g]})" if g in eia_for_gem else ""
            )
            print(
                f"EIA row {e}, GEM row {g}{alt_match}: dist={d}, weight={weights[e, g]}"
            )
    for e in missing_candidates:
        print(f"EIA row {e}: no candidates in GEM tracker")

gem_operating = set(gem[gem["Status"] == "operating"]["gem_row"])
missing_candidates = gem_operating - set(gem_rows)
if gem_shortfall or missing_candidates:
    print(f"Unmatched GEM rows:")
    for (e, g), d in sorted(zip(pairs, c_w), key=lambda x: (x[0][1], x[0][0])):
        if g in gem_shortfall and g in gem_operating:
            alt_match = (
                f" (already assigned to GEM row {gem_for_eia[e]})"
                if e in gem_for_eia
                else ""
            )
            print(
                f"GEM row {g}: matches EIA row {e}{alt_match}: dist={d}, weight={weights[e, g]}"
            )
    for g in missing_candidates:
        print(f"GEM row {g}: no matching EIA coal plants")

# check that status for matched GEM projects is online or retired
assert set(gem[gem["gem_row"].isin(eia_for_gem)]["Status"].unique()) == {
    "operating",
    "retired",
}

# calculate new retirement date for each row in EIA table, based on best matching GEM
new_retirement_year = (
    eia["eia_row"].map(gem_for_eia).map(gem.set_index("gem_row")["Planned retirement"])
)

# apply new retirement year unless it is nan (sometimes they disagree, but we assume
# gem is more up to date); then convert remaining NaNs to " " to match EIA workbook
eia["GEM Retirement Year"] = new_retirement_year.fillna(eia["Planned Retirement Year"])

# save results to PowerGenome settings
year_col = eia.columns.get_loc("Planned Retirement Year") + 1
# records will be (EIA plant ID, EIA generator ID, retirement year)
addl_retirements = []
for i, (row, plant_id, gen_id, old_year, new_year) in eia[
    [
        "eia_row",
        "Plant ID",
        "Generator ID",
        "Planned Retirement Year",
        "GEM Retirement Year",
    ]
].iterrows():
    # save any retirement years that have changed
    if new_year != old_year and not (np.isnan(new_year) and np.isnan(old_year)):
        try:
            # convert to int if possible to reduce quoting and reformatting in VS Code
            # PG will convert back
            gen_id = int(gen_id)
        except:
            pass
        new_year = None if np.isnan(new_year) else int(new_year)
        row = CommentedSeq([plant_id, gen_id, new_year])
        row.fa.set_flow_style()  # show whole record on one row
        addl_retirements.append(row)

# Replace any existing additional_retirements key with the new list
ar = read_yaml(addl_retirements_file)
# delete_yaml_keys(ar, ["additional_retirements"])
add_yaml_key(ar, ["additional_retirements"], addl_retirements)
write_yaml(ar, addl_retirements_file)
# print(f"Updated additional_retirements key in {addl_retirements_file}.")
