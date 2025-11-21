"""
Create group, metadata and profile files in RESOURCE_GROUP folder for
existing resource groups that are present in the EIA 860 data but not
in the PowerGenome resource group dataset (generally because they
were added recently and weren't in the version of EIA 860 used to make
that). These use the same profile as an equivalent new-build resource.

This script should be run after download_pg_data.py and before running
pg_to_switch.py.
"""

# %%##########################
# setup
import tempfile, shutil, os
from pathlib import Path

# avoid geopandas warning about using shapely vs. pygeos
os.environ["USE_PYGEOS"] = "0"

import pandas as pd
import geopandas as gpd
from powergenome.util import load_settings
from powergenome.generators import GeneratorClusters
from powergenome.util import (
    init_pudl_connection,
    load_settings,
    snake_case_col,
)
from download_pg_data import make_parent, gdown, unzip_if_needed
from pg_to_switch import short_fn


def read_file(file, **kwargs):
    if str(file).endswith(".csv"):
        return pd.read_csv(file, **kwargs)
    elif str(file).endswith(".parquet"):
        return pd.read_parquet(file, **kwargs)
    else:
        raise ValueError(f"Don't know how to read {short_fn(file)} into a DataFrame.")


def to_file(df, file, **kwargs):
    print(f"Updating {short_fn(file)}.")
    if str(file).endswith(".csv"):
        return df.to_csv(file, index=False, **kwargs)
    elif str(file).endswith(".parquet"):
        return df.to_parquet(file, **kwargs)
    else:
        raise ValueError(f"Don't know how to save DataFrame to {short_fn(file)}.")


# TODO: get from argv
settings_dir = "pg/settings"
print(f"Reading settings from {settings_dir}")
settings = load_settings(settings_dir)

# %%###########################
# Download original PowerGenome data files for ReEDS zones
# and apply several fixes. The files can then be uploaded to a Google Drive and
# shared for other users to download via the standard pg_data.yml / download_pg_data.py.

# Fixes applied:
# cpas/offshorewind_lcoe_ReEDS.csv:
#   - rename ipm_region_y (ReEDS zones) to ipm_region
#   - for cpas that are closest to MD or RI (p123 and p133), assign that
#     as the ipm_region (these are missing from the original CPA metadata,
#     which makes it impossible to meet offshore wind mandates in those states)
# existing_rg/existing_hydro_reeds_ba_metadata.csv:
#   - drop rows that don't have profiles in existing_hydro_reeds_ba.parquet
#     TODO: somehow find profiles for them instead
# existing_rg/existing_offshorewind_reeds_ba_metadata.csv:
#   - add dummy rows for any regions that are in existing_osw_profiles.csv
#     but not in the metadata
# existing_rg/existing_osw_profiles.csv:
#   - reduce capacity factors by a factor of 1000 to get reasonable values (0-1).

tmpdir = Path(tempfile.mkdtemp())

pg_files = {
    # PowerGenome data files for ReEDS zones from
    # https://drive.google.com/drive/folders/1SDj2XuS-GYRpsWto5i1yICOYP-xENskk
    # (shared by Greg Schivley via Slack 2025-05-13). Greg Schivley created
    # these profiles in May 2025 by applying NREL's ReV tool to Candidate
    # Project Areas (CPAs) previously identified using Princeton's longstanding
    # screening and interconnection tools. ReV converts meteorological data from
    # Wind Toolkit and NSRDB into power production profiles.
    "cpas": (
        "pg_data/resource_groups/ReEDS-cpas-patched.zip",
        "https://drive.google.com/file/d/1bZQPwEsgL11g_KWeniJ41YZhUMEpcVKW/view?usp=drive_link",
    ),
    "existing_rg": (
        "pg_data/resource_groups/existing_resource_groups-patched.zip",
        "https://drive.google.com/file/d/1ZXvHnJxftjgWzeF_HXB0uxw3l2cIQsbW/view?usp=drive_link",
    ),
    # Custom resource group data prepared by Greg Schivley for MIP project in March 2024
    # (https://dx.doi.org/10.2139/ssrn.5205762 and https://github.com/switch-model/Switch-USA-PG)
    # Outer folder is here: https://drive.google.com/drive/u/2/folders/1KBdoonCeDfvAgQ10KpwVhmlyfKi1rY5K
    # Link below is for conus-26 resource groups/conus-26-resource-groups.zip
    "mip": (
        str(tmpdir / "conus-26-resource-groups.zip"),
        "https://drive.google.com/file/d/1eIu3HxAoR4-tYcvew8AMvxWGtgWE52M0/view?usp=drive_link",
    ),
}

# Download the original files (this code is very similar to part of download_pg_data.py)
print("\nDownloading original data prepared by PowerGenome team.")
for dest, url in pg_files.values():
    print(f"\nretrieving {dest}")
    make_parent(dest)
    filename = gdown.download(url, fuzzy=True, output=dest)
    unzip_if_needed(filename)
print()

pg_dir = lambda k: Path(pg_files[k][0]).with_suffix("")

# rename existing_resource_groups_20250513 (name automatically assigned from
# folder inside zip file) to expected name ('existing_resource_groups_patched')
new_path = pg_dir("existing_rg")
if new_path.is_dir():
    shutil.rmtree(new_path)
elif new_path.is_file():  # unlikely
    new_path.unlink()
old_path = new_path.parent / "existing_resource_groups_20250513"
old_path.replace(new_path)

# cpas/offshorewind_lcoe_ReEDS.csv:
#   - rename ipm_region_y (ReEDS zones) to ipm_region
#   - for cpas that are closest to MD or RI (p123 and p133), assign that as the
#     ipm_region and metro_region (these are missing from the original CPA
#     metadata, which makes it impossible to meet offshore wind mandates in
#     those states)

print("Moving offshore wind CPAs to nearest region if it is p123 or p133.")
meta_file = pg_dir("cpas") / "offshorewind_lcoe_ReEDS.csv"
meta = read_file(meta_file)
meta = meta.drop(columns="ipm_region").rename(columns={"ipm_region_y": "ipm_region"})

# get longitude and latitude by cross-referencing MIP version of metadata
mip_meta = read_file(pg_dir("mip") / "offshorewind_lcoe_conus_26_zone.csv")
meta = meta.merge(mip_meta[["CPA_ID", "latitude", "longitude"]], how="left")
meta = gpd.GeoDataFrame(
    meta,
    geometry=gpd.points_from_xy(meta["longitude"], meta["latitude"]),
    crs="EPSG:4326",
).to_crs("EPSG:5070")

# find closest zone using shapefile
regions = gpd.read_file(
    settings["input_folder"] / "US_PCA_region" / "US_PCA_region.shp"
).to_crs("EPSG:5070")
nearest = gpd.sjoin_nearest(
    meta, regions[["region", "geometry"]], how="left", distance_col="dist_to_region"
)
# drop weirdly equidistant ones
nearest = nearest.drop_duplicates(subset="CPA_ID")
meta["nearest_region"] = meta["CPA_ID"].map(nearest.set_index("CPA_ID")["region"])

# assign nearest region if it is one of the two that are missing data
# (this is more conservative than reassigning all of them; maybe PG data
# used the landing point for tie lines instead of closest state?)
meta.loc[meta["nearest_region"].isin(["p123", "p133"]), "ipm_region"] = meta[
    "nearest_region"
]
meta["metro_region"] = meta["ipm_region"]

# save for future use
to_file(meta.drop(columns="geometry"), meta_file)

# existing_rg/existing_hydro_reeds_ba_metadata.csv:
#   - drop rows that don't have profiles in existing_hydro_reeds_ba.parquet
# TODO: somehow find profiles for them instead
print(
    "Dropping two existing hydro sites that lack time profiles (TODO: find profiles for these)"
)
hydro_profiles = read_file(pg_dir("existing_rg") / "existing_hydro_reeds_ba.parquet")
hydro_meta_file = pg_dir("existing_rg") / "existing_hydro_reeds_ba_metadata.csv"
hydro_meta = read_file(hydro_meta_file)
hydro_meta = hydro_meta.loc[hydro_meta["ipm_region"].isin(hydro_profiles.columns), :]
to_file(hydro_meta, hydro_meta_file)

# existing_rg/existing_offshorewind_reeds_ba_metadata.csv:
#   - add dummy rows for any regions that are in existing_osw_profiles.csv
#     but not in the metadata
print(
    "Adding metadata rows for two profiles of existing offshore wind that are missing "
    "from the metadata."
)
osw_meta_file = pg_dir("existing_rg") / "existing_offshorewind_reeds_ba_metadata.csv"
osw_profile_file = pg_dir("existing_rg") / "existing_osw_profiles.csv"
osw_meta = read_file(osw_meta_file)
osw_profiles = read_file(osw_profile_file)
missing = list(set(osw_profiles.columns) - set(osw_meta["ipm_region"]))
osw_meta = pd.concat([osw_meta, pd.DataFrame({"ipm_region": missing, "id": missing})])
to_file(osw_meta, osw_meta_file)

# existing_rg/existing_osw_profiles.csv:
#   - reduce capacity factors by a factor of 1000 to get reasonable values (0-1).
osw_profile_file = pg_dir("existing_rg") / "existing_osw_profiles.csv"
print(f"Correcting scale for {osw_profile_file}")
osw_profiles = read_file(osw_profile_file)
mx = osw_profiles.max(axis=0)
assert (
    (mx >= 100) & (mx <= 1000)
).all(), "Expected to find cap factor in 0-1000 range."
osw_profiles *= 0.001
to_file(osw_profiles, osw_profile_file)

# remove the temporary folder
shutil.rmtree(tmpdir)

# %%################
# for every new-build technology in every ipm_region, make sure there's an
# equivalent existing technology in that region. For any new-build technologies
# that don't already have an existing-build technology in the same ipm_region,
# this adds a synthetic existing-build resource to the metadata (csv) and
# profiles (parquet) files, based on the lowest-lcoe new-build option in that
# ipm_region.

print(f"Reading generator cluster data")

pudl_engine, pudl_out, pg_engine = init_pudl_connection(
    freq="AS",
    start_year=min(settings.get("eia_data_years")),
    end_year=max(settings.get("eia_data_years")),
    pudl_db=settings.get("PUDL_DB"),
    pg_db=settings.get("PG_DB"),
)
gc = GeneratorClusters(pudl_engine, pudl_out, pg_engine, settings)


print(
    "Adding profiles for existing resources where needed, "
    "based on new-build profiles."
)

new_build_techs = set(
    g.group["technology"] for g in gc.cluster_builder.find_groups(existing=False)
)

for tech in new_build_techs:
    if tech == "imports":  # skip, will be defined later (and not modeled as existing)
        continue
    print("")  # blank line between groups
    exist_group = gc.cluster_builder.find_groups(existing=True, technology=tech)[
        0
    ].group
    new_group = gc.cluster_builder.find_groups(existing=False, technology=tech)[0].group
    exist_md = read_file(exist_group["metadata"])
    new_md = read_file(new_group["metadata"])

    patch_new_md = new_md.query("~ipm_region.isin(@exist_md['ipm_region'])")
    # assign the lowest-lcoe CPA_ID for this region, since that is probably what was built first
    patch_cpa = patch_new_md.loc[
        patch_new_md.groupby("ipm_region")["lcoe"].idxmin(), ["ipm_region", "CPA_ID"]
    ]
    # Retrieve the site map and use that to find the Site ID, identifying the
    # matching column in the profiles parquet file.
    new_site_map = read_file(new_group["profiles"].parent / new_group["site_map"])
    patch_cpa["Site"] = patch_cpa["CPA_ID"].map(
        new_site_map.set_index("CPA_ID")["Site"]
    )
    print(
        f"Found {len(patch_cpa)} regions that have new-build {tech} but not existing."
    )
    if patch_cpa.empty:
        continue

    print(f"Creating existing {tech} profiles based on new-build profiles.")
    # build profiles and metadata for existing-type projects based on the new-type ones
    patch_profiles = read_file(
        new_group["profiles"], columns=patch_cpa["Site"].astype(str).to_list()
    ).rename(columns=patch_cpa.set_index("Site")["ipm_region"])
    patch_md = pd.DataFrame(
        {
            "ipm_region": patch_cpa["ipm_region"],
            "capacity_mw": None,
            "id": patch_cpa["ipm_region"],
            "mw": None,
        }
    )

    # add these back into the existing metadata and profile files
    exist_profiles = read_file(exist_group["profiles"])
    exist_profiles = pd.concat([exist_profiles, patch_profiles], axis=1)
    to_file(exist_profiles, exist_group["profiles"])

    exist_md = pd.concat([exist_md, patch_md], axis=0)
    to_file(exist_md, exist_group["metadata"])


# %%#############
# make an alternative to reeds_ba_tx_NARIS_avg.csv based on REFS2009 instead of NARIS
tx_file = settings["input_folder"] / "reeds_ba_tx_REFS2009_avg.csv"
ac_url = "https://github.com/NREL/ReEDS-2.0/raw/refs/heads/main/inputs/transmission/transmission_capacity_init_AC_ba_REFS2009.csv"
dc_url = "https://github.com/NREL/ReEDS-2.0/raw/refs/heads/main/inputs/transmission/transmission_capacity_init_nonAC_ba.csv"

print(f"\nCreating {tx_file} from {ac_url} and {dc_url}")
ac = pd.read_csv(ac_url)
dc = pd.read_csv(dc_url)
ac["MW"] = ac[["MW_f0", "MW_r0"]].mean(axis=1)
ac["Project(s)"] = "REFS2009 avg value"

tx = pd.concat(
    [dc[["r", "rr", "MW", "Project(s)"]], ac[["r", "rr", "MW", "Project(s)"]]]
)
tx = tx.rename(
    columns={
        "r": "region_from",
        "rr": "region_to",
        "MW": "firm_ttc_mw",
        "Project(s)": "notes",
    }
)
tx = tx.groupby(["region_from", "region_to"], as_index=False).agg(
    firm_ttc_mw=("firm_ttc_mw", "sum"), notes=("notes", lambda x: " + ".join(x))
)

to_file(tx, tx_file)
