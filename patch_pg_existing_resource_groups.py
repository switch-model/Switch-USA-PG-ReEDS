"""
Create group, metadata and profile files in RESOURCE_GROUP folder for
existing resource groups that are present in the EIA 860 data but not
in the PowerGenome resource group dataset (generally because they
were added recently and weren't in the version of EIA 860 used to make
that). These use the same profile as an equivalent new-build resource.

This script should be run after download_pg_data.py and before running
pg_to_switch.py.
"""

import pandas as pd
from powergenome.util import load_settings
from powergenome.generators import GeneratorClusters
from powergenome.util import (
    init_pudl_connection,
    load_settings,
    snake_case_col,
)
from pg_to_switch import short_fn

print("=" * 80)
print(
    "Several manual steps are needed after running this script, before using pg_to_switch.py:"
)
print(
    "TODO: remove p110 from existing_hydro_reeds_ba_metadata.csv since it isn't in existing_hydro_reeds_ba.parquet,"
)
print(
    "TODO: reduce capacity factors by a factor of 1000 in existing_resource_groups_20250513/existing_osw_profiles.csv to get reasonable values (0-1)."
)
print(
    "TODO: move data from ipm_region_y (ReEDS zones) to ipm_region (currently ipm zones) in ReEDS-cpas/offshorewind_lcoe_ReEDS.csv."
)
print(
    "TODO: add dummy rows for p128 and p131 to existing_resource_groups_20250513/existing_offshorewind_reeds_ba_metadata.csv (they are in existing_osw_profiles.csv but not in the metadata)."
)
print("=" * 80)
# NOTE: several of the operations above can be handled automatically by adding a step that
# removes rows from metadata if not present in profiles and adds dummy rows to metadata for
# existing projects if present in profiles but not in metadata (for all existing projects,
# not just those with new-build options).
# Then we could also use semi-hard-coded actions:
# - copy ipm_region_y to ipm_region if present and drop ipm_region_y
# - divide profile by 1000 if any values exceed 500 (or maybe divide by next highest
#   order of magnitude, whatever that is)


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

print(f"Reading settings and generator clusters from {settings_dir}")

settings = load_settings(settings_dir)
pudl_engine, pudl_out, pg_engine = init_pudl_connection(
    freq="AS",
    start_year=min(settings.get("eia_data_years")),
    end_year=max(settings.get("eia_data_years")),
    pudl_db=settings.get("PUDL_DB"),
    pg_db=settings.get("PG_DB"),
)
gc = GeneratorClusters(pudl_engine, pudl_out, pg_engine, settings)

# for every new-build technology in every ipm_region, make sure there's an
# equivalent existing technology in that region. For any new-build technologies
# that don't already have an existing-build technology in the same ipm_region,
# this adds a synthetic existing-build resource to the metadata (csv) profiles
# (parquet) files, based on the lowest-lcoe new-build option in that ipm_region.

new_build_techs = set(
    g.group["technology"] for g in gc.cluster_builder.find_groups(existing=False)
)

for tech in new_build_techs:
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

    # add these back into the existing metadata and profiles files
    exist_profiles = read_file(exist_group["profiles"])
    exist_profiles = pd.concat([exist_profiles, patch_profiles], axis=1)
    to_file(exist_profiles, exist_group["profiles"])

    exist_md = pd.concat([exist_md, patch_md], axis=0)
    to_file(exist_md, exist_group["metadata"])


# make an alternative to reeds_ba_tx_NARIS_avg.csv based on REFS2009 instead of NARIS
tx_file = settings["input_folder"] / "reeds_ba_tx_REFS2009_avg.csv"
ac_url = "https://github.com/NREL/ReEDS-2.0/raw/refs/heads/main/inputs/transmission/transmission_capacity_init_AC_ba_REFS2009.csv"
dc_url = "https://github.com/NREL/ReEDS-2.0/raw/refs/heads/main/inputs/transmission/transmission_capacity_init_nonAC_ba.csv"

print(f"Creating {tx_file} from {ac_url} and {dc_url}")
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
