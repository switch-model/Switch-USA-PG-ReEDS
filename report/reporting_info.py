"""
Group definitions used by make_load_balance_graphs.py and
make_capacity_production_tables.py
"""

import io, textwrap
from pathlib import Path

import pandas as pd

# sometimes used for comparisons
historical_year = 2025

model_year = 2030

ce_in_dir = Path(f"in/{model_year}/ce/")
ra_in_dir = Path(f"in/{model_year}/ra/")
ce_out_dir = Path(f"out/{model_year}/ce/")
ra_out_dir = Path(f"out/{model_year}/ra/")
# directory where summary graphs and tables will be stored
summary_out_dir = Path("out")

fonts_dir = Path(__file__).parent / "fonts"

logo_file = Path(__file__).parent / "logo.png"
logo_zoom = 0.06

# same list as in adjust/make_split_models.py or adjust/add_extreme_days.py
timepoint_files = {
    # "hydro_timepoints.csv": "timepoint_id",  # not used for this project
    "loads.csv": "TIMEPOINT",
    "variable_capacity_factors.csv": "TIMEPOINT",
    "water_node_tp_flows.csv": "TIMEPOINTS",
    "dr_data.csv": "TIMEPOINT",
    "ee_data.csv": "TIMEPOINT",
}


scen_names = ["high_fossil", "clean"]

gen_tech_names = {
    "Batteries": "Storage",
    "Biomass": "Other",
    "Conventional Hydroelectric": "Hydro",
    "Natural Gas Fired Combined Cycle": "Gas CCGT",
    "Natural Gas Fired Combustion Turbine": "Gas CT",
    "Onshore Wind Turbine": "Onshore Wind",
    "Other_peaker": "Other",
    "Petroleum Liquids": "Other",
    "Small Hydroelectric": "Hydro",
    "Solar Photovoltaic": "Large Solar",
    "Conventional Steam Coal": "Coal",
    "Geothermal": "Other",
    "Hydroelectric Pumped Storage": "Storage",
    "Waste Heat": "Other",
    "Nuclear": "Nuclear",
    "Nuclear_Nuclear - Large_Moderate": "Nuclear",
    "Offshore Wind Turbine": "Offshore Wind",
    "distributed_generation": "Dist. Solar",
    "NaturalGas_1-on-1 Combined Cycle (H-Frame)_Moderate": "Gas CCGT",
    "NaturalGas_Combustion Turbine (F-Frame)_Moderate": "Gas CT",
    "Utility-Scale Battery Storage_Lithium Ion_Advanced": "Storage",
    "UtilityPV_Class1_Moderate": "Large Solar",
    "LandbasedWind_Class3_Moderate": "Onshore Wind",
    "OffShoreWind_Class3_Moderate_fixed_1": "Offshore Wind",
    "OffShoreWind_Class3_Moderate_fixed_0": "Offshore Wind",
    "OffShoreWind_Class12_Moderate_floating_0": "Offshore Wind",
    "LandbasedWind_Class3_Conservative": "Onshore Wind",
    "OffShoreWind_Class3_Conservative_fixed_1": "Offshore Wind",
    "OffShoreWind_Class3_Conservative_fixed_0": "Offshore Wind",
    "OffShoreWind_Class12_Conservative_floating_0": "Offshore Wind",
    "Imports_base_base": "Imports",
    "load_growth": "Other",
}

rto_groups = {
    "All U.S.": [f"rto{n}" for n in range(1, 18 + 1)],
    "PJM": ["rto7", "rto12"],
    "CAISO": ["rto4"],
    "MISO": ["rto6", "rto11", "rto13"],
    "SPP": ["rto8"],
    "ERCOT": ["rto10"],
    "Non-RTO West": ["rto1", "rto2", "rto3", "rto5"],
    "Non-RTO South": ["rto9", "rto14", "rto15", "rto16"],
    "NYISO": ["rto17"],
    "ISONE": ["rto18"],
}

# data from ReEDS shapefile
region_info = pd.read_csv(io.StringIO(textwrap.dedent("""\
    region,interconnect,country,custreg,rto,st
    p1,wscc,usa,Pacific,rto1,wa
    p2,wscc,usa,Pacific,rto1,wa
    p3,wscc,usa,Pacific,rto1,wa
    p4,wscc,usa,Pacific,rto1,wa
    p5,wscc,usa,Pacific,rto1,or
    p6,wscc,usa,Pacific,rto1,or
    p7,wscc,usa,Pacific,rto1,or
    p8,wscc,usa,Pacific,rto2,ca
    p9,wscc,usa,Pacific,rto4,ca
    p10,wscc,usa,Pacific,rto4,ca
    p11,wscc,usa,Pacific,rto4,ca
    p12,wscc,usa,Pacific,rto2,nv
    p13,wscc,usa,Southwest,rto3,nv
    p14,wscc,usa,Southwest,rto1,id
    p15,wscc,usa,Mountain,rto1,id
    p16,wscc,usa,Mountain,rto1,id
    p17,wscc,usa,Southwest,rto1,mt
    p18,wscc,usa,Mountain,rto2,mt
    p19,wscc,usa,Mountain,rto2,mt
    p20,wscc,usa,Mountain,rto2,mt
    p21,wscc,usa,Mountain,rto2,wy
    p22,wscc,usa,Mountain,rto2,wy
    p23,wscc,usa,Mountain,rto5,wy
    p24,wscc,usa,Mountain,rto5,wy
    p25,wscc,usa,Mountain,rto2,ut
    p26,wscc,usa,Southwest,rto2,ut
    p27,wscc,usa,Southwest,rto3,az
    p28,wscc,usa,Southwest,rto3,az
    p29,wscc,usa,Southwest,rto3,az
    p30,wscc,usa,Southwest,rto3,az
    p31,wscc,usa,Mountain,rto3,nm
    p32,wscc,usa,South Central,rto5,sd
    p33,wscc,usa,Mountain,rto5,co
    p34,wscc,usa,Mountain,rto5,co
    p35,eastern,usa,Great Plains,rto11,mt
    p36,eastern,usa,Mountain,rto11,nd
    p37,eastern,usa,Great Plains,rto11,nd
    p38,eastern,usa,Great Plains,rto11,sd
    p39,eastern,usa,Great Plains,rto8,ne
    p40,eastern,usa,Great Plains,rto8,ne
    p41,eastern,usa,Great Plains,rto8,ne
    p42,eastern,usa,Great Plains,rto11,mn
    p43,eastern,usa,Great Plains,rto11,mn
    p44,eastern,usa,Great Plains,rto11,mn
    p45,eastern,usa,Great Lakes,rto6,ia
    p46,eastern,usa,Great Plains,rto11,wi
    p47,eastern,usa,Southwest,rto8,nm
    p48,eastern,usa,South Central,rto8,tx
    p49,eastern,usa,South Central,rto8,ok
    p50,eastern,usa,South Central,rto8,ok
    p51,eastern,usa,South Central,rto8,ok
    p52,eastern,usa,Great Plains,rto8,ks
    p53,eastern,usa,Great Plains,rto8,ks
    p54,eastern,usa,Great Plains,rto8,mo
    p55,eastern,usa,Great Plains,rto8,mo
    p56,eastern,usa,South Central,rto8,ar
    p57,eastern,usa,South Central,rto8,tx
    p58,eastern,usa,South Central,rto13,la
    p59,wscc,usa,Southwest,rto3,tx
    p60,texas,usa,South Central,rto10,tx
    p61,texas,usa,South Central,rto10,tx
    p62,texas,usa,South Central,rto10,tx
    p63,texas,usa,South Central,rto10,tx
    p64,texas,usa,South Central,rto10,tx
    p65,texas,usa,South Central,rto10,tx
    p66,eastern,usa,South Central,rto13,tx
    p67,texas,usa,South Central,rto10,tx
    p68,eastern,usa,Great Plains,rto11,mn
    p69,eastern,usa,Great Plains,rto6,ia
    p70,eastern,usa,Great Plains,rto6,ia
    p71,eastern,usa,Great Plains,rto6,mo
    p72,eastern,usa,Great Plains,rto6,mo
    p73,eastern,usa,Great Plains,rto13,mo
    p74,eastern,usa,Great Lakes,rto6,mi
    p75,eastern,usa,Great Lakes,rto6,wi
    p76,eastern,usa,Great Lakes,rto6,wi
    p77,eastern,usa,Great Lakes,rto11,wi
    p78,eastern,usa,Great Lakes,rto6,wi
    p79,eastern,usa,Great Lakes,rto6,wi
    p80,eastern,usa,Great Lakes,rto7,il
    p81,eastern,usa,Great Lakes,rto6,il
    p82,eastern,usa,Great Lakes,rto6,il
    p83,eastern,usa,Great Lakes,rto6,il
    p84,eastern,usa,Great Plains,rto13,mo
    p85,eastern,usa,South Central,rto13,ar
    p86,eastern,usa,South Central,rto13,la
    p87,eastern,usa,Southeast,rto13,ms
    p88,eastern,usa,Southeast,rto14,ms
    p89,eastern,usa,Southeast,rto14,al
    p90,eastern,usa,Southeast,rto9,al
    p91,eastern,usa,Southeast,rto9,fl
    p92,eastern,usa,Southeast,rto14,tn
    p93,eastern,usa,Southeast,rto14,ky
    p94,eastern,usa,Southeast,rto9,ga
    p95,eastern,usa,Southeast,rto15,sc
    p96,eastern,usa,Southeast,rto15,sc
    p97,eastern,usa,Southeast,rto15,nc
    p98,eastern,usa,Southeast,rto15,nc
    p99,eastern,usa,Southeast,rto7,va
    p100,eastern,usa,Southeast,rto7,va
    p101,eastern,usa,Southeast,rto16,fl
    p102,eastern,usa,Southeast,rto16,fl
    p103,eastern,usa,Great Lakes,rto6,mi
    p104,eastern,usa,Great Lakes,rto7,mi
    p105,eastern,usa,Great Lakes,rto6,in
    p106,eastern,usa,Great Lakes,rto6,in
    p107,eastern,usa,Great Lakes,rto6,in
    p108,eastern,usa,Southeast,rto6,ky
    p109,eastern,usa,Southeast,rto7,ky
    p110,eastern,usa,Southeast,rto7,ky
    p111,eastern,usa,Great Lakes,rto7,oh
    p112,eastern,usa,Great Lakes,rto7,oh
    p113,eastern,usa,Great Lakes,rto7,oh
    p114,eastern,usa,Great Lakes,rto7,oh
    p115,eastern,usa,Northeast,rto7,pa
    p116,eastern,usa,Southeast,rto7,wv
    p117,eastern,usa,Southeast,rto7,wv
    p118,eastern,usa,Southeast,rto7,va
    p119,eastern,usa,Northeast,rto12,pa
    p120,eastern,usa,Northeast,rto7,pa
    p121,eastern,usa,Southeast,rto7,md
    p122,eastern,usa,Northeast,rto12,pa
    p123,eastern,usa,Southeast,rto12,md
    p124,eastern,usa,Southeast,rto12,va
    p125,eastern,usa,Southeast,rto12,de
    p126,eastern,usa,Northeast,rto12,nj
    p127,eastern,usa,Northeast,rto17,ny
    p128,eastern,usa,Northeast,rto17,ny
    p129,eastern,usa,Northeast,rto18,vt
    p130,eastern,usa,Northeast,rto18,nh
    p131,eastern,usa,Northeast,rto18,ma
    p132,eastern,usa,Northeast,rto18,ct
    p133,eastern,usa,Northeast,rto18,ri
    p134,eastern,usa,Northeast,rto18,me
    """).strip()))

zone_groups = {
    g: region_info.query("rto.isin(@rto_list)")["region"].to_list()
    for g, rto_list in rto_groups.items()
}

supply_colors = {
    "Nuclear": "#93358F",
    "Coal": "#000000",
    "Other": "#AB7942",
    "Gas CCGT": "#AEAEAE",
    "Gas CT": "#707070",
    "Hydro": "#0B76A0",
    "Storage": "#96DCF8",
    "Imports": "#FF0000",
    "Onshore Wind": "#47D45A",
    "Offshore Wind": "#4EA72E",
    "Large Solar": "#FFC000",
    "Dist. Solar": "#E97132",
}
demand_patterns = {
    "Base Demand": dict(color="black", linewidth=0.75, linestyle="dotted"),
    "Modified Demand": dict(color="black", linewidth=0.75, linestyle="solid"),
    # "Exports": dict(color="black", linewidth=2, linestyle="solid"),
    # "PRM": dict(color="red", linewidth=2, linestyle="dotted"),
}

supply_cols = list(supply_colors.keys())
demand_cols = list(demand_patterns.keys())


def assert_all_in(grp1, grp2, msg="missing values"):
    # raise an error if any members of grp1 are not in grp2
    missing = set(grp1) - set(grp2)
    if missing:
        raise AssertionError(f"{msg}: {missing}")


assert_all_in(
    gen_tech_names.values(), supply_cols, "unexpected graph labels in gen_tech_names"
)
