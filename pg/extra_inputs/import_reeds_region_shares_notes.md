import_reeds_region_shares.csv shows the allocation of each (international BA) -
(US BA) pair to individual ReEDS zones, where the BAs are identified by EIA codes
as done in the EIA 930 data. This can be used to allocate international interchange
reported in EIA Form 930 to specific ReEDS zones as the end points.

Strategy to make import_reeds_region_shares.csv:
- query the EIA 930 data to find the international and US BA pairs for imports
  and exports in 2024 (see make_study_loads.py)
  - list the EIA BA codes for those foreign-US pairs below
  - eia_pairs = interchange.query(f"neighbor_region.isin({'Canada',
    'Mexico'})")[['ba_code',
    'balancing_authority_code_adjacent_eia']].dropna().drop_duplicates().sort_values(['ba_code',
    'balancing_authority_code_adjacent_eia'], axis=0)
- in QGIS, load these sources (see gis dir)
  - EIA BA shapefile from
    https://atlas.eia.gov/datasets/09550598922b429ca9f06b9a067257bd_255/explore
  - Open Infrastructure Map  (based on OpenStreetMap) vector tiles from
    https://openinframap.org/tiles/{z}/{x}/{y}.pbf (shows all transmission lines
    in N. America)
    - can be helpful to put "AND line IS NOT NULL" as an additional filter
      condition for line features in the symbology tab, to filter out
      communication lines and pipelines (but may miss cables?)
  - Canadian BA regions from georeferenced
    https://www.nerc.com/AboutNERC/keyplayers/PublishingImages/BA%20Bubble%20Map%2020191106.tif
    available as "NERC Balancing Authority Areas (As of October 2019)" on
    https://www.nerc.com/AboutNERC/keyplayers/Pages/default.aspx 
  - ReEDS regions from shapefile in this directory (also US_PCA in
    https://github.com/NREL/ReEDS-2.0/tree/main/inputs/shapefiles)
  - see gis dir for these
  - other useful information:
    - CA-US line ratings and ID's from
      https://www.cer-rec.gc.ca/en/data-analysis/facilities-we-regulate/international-power-lines-dashboard/
    - EIA BA code - named region crosswalk at
      https://www.eia.gov/electricity/930-content/EIA930_Reference_Tables.xlsx
- in QGIS, zoom in on the border regions between each foreign BA (shown on the
  NERC TIF image) and US BA from the list below
  - can be helpful to use the EIA US BA shapefile and turn off all BAs except
    the relevant one since they overlap
- for each line, use the identify tool on the Open Infra Map to find the voltage
  rating, and use the identify tool on the ReEDS region layer to identify the
  US ReEDS region the line ends in. Add this data to the table below.
- create import_reeds_region_shares.csv by hand from the line voltage table by
  prorating between ReEDS regions by line voltage (i.e., assume all lines have
  similar current, which seems true for the Canadian lines I got MW ratings
  for). This gives an estimate of the share of interchange on each EIA BA pair
  that goes into each ReEDS region.

# Other data sources considered but dropped:

Get ReEDS pairs from https://github.com/NREL/ReEDS-2.0/blob/main/inputs/shapefiles/r_rr_lines_to_25_nearest_neighbors.csv? But that seems too big, seems to be all 25 nearest neighbors for each zone.

Use ready-made canada exports from https://github.com/NREL/ReEDS-2.0/tree/main/inputs/canada_imports ? Doesn't give us enough control, and may lock us into an incomplete version of ReEDS's treatment? (exports to Canada treated as loads, imports treated as hydro plants, but we don't have the details on those hydro plants)

One of the files like this (but doesn't seem to cover non-US): https://github.com/NREL/ReEDS-2.0/blob/main/inputs/transmission/transmission_capacity_init_AC_ba_NARIS2024.csv

Possibly use Figure 10 of https://docs.nrel.gov/docs/fy13osti/56724.pdf for Canadian
transmission lines in ReEDS zones? This extension included Mexico but didn't show a map,
but maybe there is a map in their Mexico reports (https://docs.nrel.gov/docs/fy15osti/63797.pdf)?

Maybe these help? 
https://openinframap.org/#4.83/48.32/-111.15
https://atlas.eia.gov/datasets/border-crossings-electricity/
https://en.wikipedia.org/wiki/North_American_power_transmission_grid
https://upload.wikimedia.org/wikipedia/commons/b/bd/High_voltage_power_grid_of_the_United_States.webp
https://www.arcgis.com/apps/mapviewer/index.html?layers=d4090758322c4d32a4cd002ffaa0aa12
https://www.arcgis.com/home/item.html?id=d4090758322c4d32a4cd002ffaa0aa12
https://www.cer-rec.gc.ca/en/data-analysis/facilities-we-regulate/international-power-lines-dashboard/
https://community.esri.com/t5/electric-questions/open-street-map-showing-transmission-line/td-p/188195
https://www.arcgis.com/home/item.html?id=4b6f78254c2b48338d5cf155b204ad40
https://www.arcgis.com/apps/mapviewer/index.html?layers=4b6f78254c2b48338d5cf155b204ad40

Generally OpenStreetMap seems the most promising source for georeferenced cross-border lines. But the gc.ca dashboard has some MW ratings.

By inspection of layers and sources discussed at top of this file, created the initial table below.

note:
IESO - Ontario Indep. ESO
MHEB - Manitoba Hydro
HQT - Hydro Quebec
NBSO - New Brunswick System Operator
SPC - Saskatchewan Power Corporation

from,to,region,line_kv,name,cer_name,cer_mw,start_loc,end_loc

BCHA,BPAT,p1,500,Ingledow - Custer 500KV #1,EC-III-4,700,Ingledow BC,Blaine WA
BCHA,BPAT,p1,500,Ingledow - Custer 500KV #2,EC-III-12,700,Ingledow BC,Blaine WA
BCHA,BPAT,p3,230,Boundary-Waneta No 1,EC-III-1,300,Waneta GS,Spokane WA
BCHA,BPAT,p3,230,Boundary-Nelway No 1,EC-III-10,380,Cranbrook BC,Boundary WA

AESO,NWMT,p18,230,Montana-Alberta Tie,EP-301,300,Lethbridge AB,Great Falls MT

SPC,SWPP,p36,230,Tioga - Boundary Dam SPC,

MHEB,MISO,p37,230,Glenboro - Rugby 230kV,
MHEB,MISO,p42,500,Manitoba-Minnesota Transmission Line
MHEB,MISO,p42,500,Riel - Roseau County 500KV
MHEB,MISO,p42,230,no name,

IESO,MISO,p42,115,no name,
IESO,MISO,p103,69,no name,
IESO,MISO,p103,69,no name,
IESO,MISO,p103,230,no name,
IESO,MISO,p103,230,Saint Clair - Lambton,
IESO,MISO,p103,345,Saint Clair - Lambton 345kV,
IESO,MISO,p103,230,no name,

IESO,NYIS,p127,many

HQT,NYIS,p127,many

HQT,ISNE,p129,120,Bedford Qc - Highgate Vt - 120kV
HQT,ISNE,p129,115,Newport - Poste de Stanstead 115kV transmission line
HQT,ISNE,p129,450,HVDC Quebec - New England Transmission Line
HQT,ISNE,p134,320,Appalaches-Maine HVDC Interconnection

# next one seems to actually be Maine Public Service Company (MPS), mainly served by NBSO
NBSO,ISNE,p134,several

CEN,CISO,p11,all

CEN,ERCO,p61,138,no name
CEN,ERCO,p65,138,no name
CEN,ERCO,p65,230,no name
CEN,ERCO,p65,138,no name
CEN,ERCO,p65,138,no name
CEN,ERCO,p65,138,no name
CEN,ERCO,p65,230,no name
CEN,ERCO,p65,138,no name
CEN,ERCO,p65,69,no name

CFE doesn't show up as a separate entity on maps and appears to have been replaced by CEN. 
We treat it as a copy of CEN when creating import_reeds_region_shares.csv.
