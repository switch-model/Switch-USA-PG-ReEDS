"""
In 2023-12, PUDL updated their database schema, breaking compatibility with
current versions of PowerGenome as of 2025-08 (powergenome 0.7.0 and earlier).

See https://catalystcoop-pudl.readthedocs.io/en/latest/release_notes.html#v2023-12-01

This script creates a pudl database with tables needed by PowerGenome and
pg_to_switch.py, using the pre-2023-12 schema with post-2023-12 data. This makes
it possible to run PowerGenome with newer pudl/EIA data.

To use this script, download the files noted below
"""

import pandas as pd
import sqlite3, pprint, os, re

# PUDL data from 2025-08, using the post-2023-12 schema.
# from https://zenodo.org/records/16878930
pudl_db = "/tmp/pudl.sqlite"

# database that will hold new pudl data in the pre-2023-12 schema;
# will be created if doesn't exist already
new_pg_db = "pg_data/PUDL Data/pudl.2025_08.sqlite"

pudl_conn = sqlite3.connect(pudl_db)
os.makedirs(os.path.dirname(new_pg_db), exist_ok=True)
if os.path.exists(new_pg_db):
    os.remove(new_pg_db)
new_pg_conn = sqlite3.connect(new_pg_db)

# table_map below was found by tracing all table access by pg_to_switch.py, then
# looking up the matching table in the mapping below.

# Official PUDL mapping from old to new schema:
# https://docs.google.com/spreadsheets/d/1RBuKl_xKzRSLgRM7GIZbc5zUYieWFE20cXumWuv5njo/edit?gid=1126117325#gid=1126117325
# Old data dictionary:
# https://catalystcoop-pudl.readthedocs.io/en/v2022.11.30/data_dictionaries/pudl_db.html
# Current data dictionary:
# https://catalystcoop-pudl.readthedocs.io/en/v2025.8.0/data_dictionaries/pudl_db.html
# Notes on the 2023-12 schema update:
# https://catalystcoop-pudl.readthedocs.io/en/latest/release_notes.html#v2023-12-01

table_map = [
    # (old_pudl_name, new_pudl_name)
    ("plants_eia", "core_pudl__assn_eia_pudl_plants"),
    ("utilities_eia", "core_pudl__assn_eia_pudl_utilities"),
    ("plants_entity_eia", "core_eia__entity_plants"),
    ("utilities_entity_eia", "core_eia__entity_utilities"),
    ("boilers_entity_eia", "core_eia__entity_boilers"),
    ("generators_entity_eia", "core_eia__entity_generators"),
    ("generators_eia860", "core_eia860__scd_generators"),
    ("utilities_eia860", "core_eia860__scd_utilities"),
    ("boiler_generator_assn_eia860", "core_eia860__assn_boiler_generator"),
    ("boiler_fuel_eia923", "core_eia923__monthly_boiler_fuel"),
    ("plants_eia860", "core_eia860__scd_plants"),
    ("generation_eia923", "core_eia923__monthly_generation"),
    ("generation_fuel_eia923", "core_eia923__monthly_generation_fuel"),
    ("generation_fuel_nuclear_eia923", "core_eia923__monthly_generation_fuel_nuclear"),
]

# columns to change back to their previous name
rename_cols = [
    # table, current name (new pudl), fixed name (old pudl)
    # reported in data dictionary, but not actually true:
    # ('generators_entity_eia', 'generator_operating_date', 'operating_date')
    (
        "generators_eia860",
        "planned_generator_retirement_date",
        "planned_retirement_date",
    )
]


def copy_table(pudl_table, pg_table, cur):
    """
    Copy `pudl_table` from pudl schema of cursor `cur` to `pg_table` in
    main schema of cursor `cur`.
    """
    cur = new_pg_conn.cursor()
    # get table creation command, including data types and indexes
    cur.execute(
        """
        SELECT sql
        FROM pudl.sqlite_master
        WHERE type='table' AND name=?
    """,
        (pudl_table,),
    )
    row = cur.fetchone()
    if row is None:
        raise ValueError(f"Table {pudl_table} not found in {pudl_db}")
    else:
        create_sql = row[0]

    # change table name, drop foreign keys and create table in main schema
    create_sql = create_sql.replace(pudl_table, pg_table, 1)
    # drop foreign key constraints since they use the old table names and we
    # don't need them anyway
    # note: the regex is tricky and fragile; may make more sense to skip
    # this part, then drop any constraints (incl foreign keys) afterward
    create_sql = re.sub(
        r"(,\s*\n\s*)?CONSTRAINT\b.*?\([^)]+\)(?=,|\)|\n)",
        "",
        create_sql,
        flags=re.IGNORECASE,
    )
    cur.execute(f"DROP TABLE IF EXISTS {pg_table}")
    cur.execute(create_sql)

    # copy data across
    cur.execute(f"INSERT INTO {pg_table} SELECT * FROM pudl.{pudl_table}")

    # # copy indexes (if any)
    # cur.execute(
    #     """
    #     SELECT name, sql
    #     FROM pudl.sqlite_master
    #     WHERE type='index' AND tbl_name=? AND sql IS NOT NULL
    # """,
    #     (pudl_table,),
    # )
    # for idx_name, idx_sql in cur.fetchall():
    #     # Rename the index to avoid name conflicts
    #     new_idx_name = f"{pg_table}_{idx_name}"
    #     new_idx_sql = idx_sql.replace(idx_name, new_idx_name, 1)
    #     new_idx_sql = new_idx_sql.replace(pudl_table, pg_table, 1)
    #     cur.execute(new_idx_sql)


cur = new_pg_conn.cursor()
cur.execute(f"ATTACH DATABASE ? AS pudl", (pudl_db,))

for pg_table, pudl_table in table_map:
    print(f"Copying pudl {pudl_table} to pg {pg_table}.")
    copy_table(pudl_table, pg_table, cur)

cur.close()
new_pg_conn.commit()

# rename columns to their old name if needed
cur = new_pg_conn.cursor()
for table, cur_name, new_name in rename_cols:
    cur.execute(
        f"""
        ALTER TABLE {table}
        RENAME COLUMN {cur_name} TO {new_name};
        """
    )
cur.close()
new_pg_conn.commit()

print("Finished copying tables.")


def report_signatures():
    """
    This function checks the table signatures (names, column names) in old and
    new pudl databases, to support creation of a cross-schema mapping. It is
    obsolete, since clearer and more authoritative info is available on the pudl
    website (referenced above), and that was used to make the final mapping.
    """
    # pre-2023-12 data, from
    # https://drive.google.com/drive/folders/1z9BdvbwgpS5QjPTrcgyFZJUb-eN2vebu
    pg_db = "pg_data/PUDL Data/pudl.sqlite"
    pg_conn = sqlite3.connect(pg_db)

    dfs = []
    # read each table name and list of columns in the table
    for conn in [pg_conn, pudl_conn]:
        df = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table'", conn
        ).set_index("name")
        for name in df.index:
            cols = set(pd.read_sql_query(f"PRAGMA table_info({name})", conn)["name"])
            df.loc[name, "cols"] = cols
        dfs.append(df)

    pg_tables, pudl_tables = dfs

    # identify all tables that contain the same columns (plus possibly others)
    def pudl_lookup(cols):
        matches = []
        for name, (pudl_cols,) in pudl_tables.iterrows():
            if cols.issubset(pudl_cols):
                matches.append(name)
        return matches

    mapping = []
    for name, (cols, *_) in pg_tables.iterrows():
        match_tables = pudl_lookup(cols) or [""]
        if len(match_tables) > 1:
            # check if the short version of the PG table's name is
            # embedded in the PUDL table's name
            short_name = name
            for tag in ["eia", "ferc1", "aer", "eia860", "eia923"]:
                short_name = short_name.replace("_" + tag, "")
            reversed_short_name = "_".join(reversed(short_name.split("_")))
            embedded = [
                t for t in match_tables if short_name in t or reversed_short_name in t
            ]
            if embedded:
                match_tables = embedded

        pg_tables.loc[name, "pudl_table"] = ", ".join(match_tables)
        mapping.append((name,) + tuple(match_tables))
        # for pudl_name in match_tables:
        #     mapping.append((name, pudl_name))

    print("tentative mapping:")
    pprint.pprint(mapping)


# # start with tentative map from report_signatures()
# # generally assign core_*** instead of out_*** when multiple match
# # generally *_eia860 becomes core_eia860__scd_* (could also be
# # out_eia__yearly_* but that seems overprocessed for what PG expects)
# # *_eia923 -> core_eia923__monthly_* (could also be out_eia923__*)
# # when no match found:
# # run % q "ATTACH '/Users/matthiasfripp/Library/CloudStorage/OneDrive-EnergyInnovation/Analysis/Electricity-Shared/Load-Growth/Switch-USA-PG/pg_data/PUDL Data/pudl.sqlite' (TYPE sqlite); USE pudl; describe (select * from generators_entity_eia);"
# # to see which columns are in each pg table, then look up the more obscure ones in https://catalystcoop-pudl.readthedocs.io/en/latest/data_dictionaries/pudl_db.html
# table_map = [
#     ("balancing_authorities_eia", "core_eia__codes_balancing_authorities"),
#     ("boiler_generator_assn_types_eia", "core_eia__codes_boiler_generator_assn_types"),
#     ("coalmine_types_eia", "core_eia__codes_coalmine_types"),
#     ("contract_types_eia", "core_eia__codes_contract_types"),
#     ("data_maturities", "core_pudl__codes_data_maturities"),
#     ("energy_sources_eia", "core_eia__codes_energy_sources"),
#     ("ferc_accounts", "core_ferc__codes_accounts"),
#     ("ferc_depreciation_lines", "core_ferc__codes_accounts"),
#     ("fuel_receipts_costs_aggs_eia", "core_eia__yearly_fuel_receipts_costs_aggs"),
#     ("fuel_transportation_modes_eia", "core_eia__codes_fuel_transportation_modes"),
#     ("fuel_types_aer_eia", "core_eia__codes_fuel_types_agg"),
#     ("momentary_interruptions_eia", "core_eia__codes_momentary_interruptions"),
#     ("operational_status_eia", "core_eia__codes_operational_status"),
#     ("plants_entity_eia", "core_eia__entity_plants"),
#     ("plants_pudl", "core_pudl__entity_plants_pudl"),
#     ("political_subdivisions", "core_pudl__codes_subdivisions"),
#     ("power_purchase_types_ferc1", "core_ferc1__codes_power_purchase_types"),
#     ("prime_movers_eia", "core_eia__codes_prime_movers"),
#     ("reporting_frequencies_eia", "core_eia__codes_reporting_frequencies"),
#     ("sector_consolidated_eia", "core_eia__codes_sector_consolidated"),
#     ("steam_plant_types_eia", "core_eia__codes_steam_plant_types"),
#     ("utilities_entity_eia", "core_eia__entity_utilities"),
#     ("utilities_pudl", "core_pudl__entity_utilities_pudl"),
#     ("boilers_entity_eia", "core_eia923__monthly_boiler_fuel"),
#     ("coalmine_eia923", "core_eia923__entity_coalmine"),
#     ("generation_fuel_eia923", "core_eia923__monthly_generation_fuel"),
#     ("generation_fuel_nuclear_eia923", "core_eia923__monthly_generation_fuel_nuclear"),
#     ("generators_entity_eia", "core_eia__entity_generators"),
#     # ("plants_eia", "core_pudl__assn_eia_pudl_plants"),
#     # ("plants_eia", "out_eia__yearly_plants"),
#     ("utilities_eia", "core_pudl__assn_eia_pudl_utilities"),
#     ("utilities_eia860", "core_eia860__scd_utilities"),
#     # ("utilities_eia860", "out_eia__yearly_utilities"),
#     ("utilities_ferc1", "core_pudl__assn_ferc1_pudl_utilities"),
#     ("utility_plant_assn", "core_pudl__assn_utilities_plants"),
#     ("boiler_fuel_eia923", "core_eia923__monthly_boiler_fuel"),
#     # ("boiler_fuel_eia923", "out_eia923__boiler_fuel"),
#     ("epacamd_eia", "core_epa__assn_eia_epacamd"),
#     ("fuel_receipts_costs_eia923", "core_eia923__monthly_fuel_receipts_costs"),
#     ("generation_eia923", "core_eia923__monthly_generation"),
#     # ("generation_eia923", "out_eia923__generation"),
#     ("plant_in_service_ferc1", "core_ferc1__yearly_plant_in_service_sched204"),
#     ("plants_eia860", "core_eia860__scd_plants"),
#     # ("plants_eia860", "out_eia__yearly_plants"),
#     ("plants_ferc1", "core_pudl__assn_ferc1_pudl_plants"),
#     (
#         "purchased_power_ferc1",
#         "core_ferc1__yearly_purchased_power_and_exchanges_sched326",
#     ),
#     ("utilities_ferc1_dbf", "core_pudl__assn_ferc1_dbf_pudl_utilities"),
#     ("utilities_ferc1_xbrl", "core_pudl__assn_ferc1_xbrl_pudl_utilities"),
#     ("fuel_ferc1", "core_ferc1__yearly_steam_plants_fuel_sched402"),
#     # next version would bring in latitude and longitude, which aren't expected
#     # ("generators_eia860", "out_eia860__yearly_generators"),
#     # next version is missing operating_date
#     ("generators_eia860", "core_eia860__scd_generators"),
#     ("plants_hydro_ferc1", "core_ferc1__yearly_hydroelectric_plants_sched406"),
#     (
#         "plants_pumped_storage_ferc1",
#         "core_ferc1__yearly_pumped_storage_plants_sched408",
#     ),
#     ("plants_small_ferc1", "core_ferc1__yearly_small_plants_sched410"),
#     ("plants_steam_ferc1", "core_ferc1__yearly_steam_plants_sched402"),
#     # ("plants_steam_ferc1", "out_ferc1__yearly_steam_plants_sched402"),
#     ("boiler_generator_assn_eia860", "core_eia860__assn_boiler_generator"),
#     ("ownership_eia860", "core_eia860__scd_ownership"),
#     # ("ownership_eia860", "out_eia860__yearly_ownership"),
# ]
