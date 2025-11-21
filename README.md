# Introduction

The sections below show how to use this repository. Note that this will take
about 25 GB of storage on your computer before you start generating model
inputs. Most of this is used by the PowerGenome input files that will be
stored in the `pg_data` directory (9.2 GB).

# Install VS Code and Python Extensions

We assume you are using the Visual Studio Code (VS Code) text editor to view and
edit code and data files and run Switch. You can use a different text editor
(and terminal app) if you like, but it should be capable of doing
programming-oriented tasks, like quickly adjusting the indentation of many lines
in a text file. If you prefer, you can also open the .csv data files directly in
your spreadsheet software instead of using VS Code.

Download and install the VS Code text editor from https://code.visualstudio.com/.

If you need more information on installing VS Code, see
https://code.visualstudio.com/docs/setup/setup-overview. (On a Mac you may need
to double-click on the downloaded zip file to uncompress it, then use the Finder
to move the “Visual Studio Code” app from your download folder to your
Applications folder.)

If you'd like a quick introduction to VS Code, see
https://code.visualstudio.com/docs.

Launch Visual Studio Code from the Start menu (Windows) or Applications folder
(Mac). You can choose a color theme and/or work your way through the “Get
Started” steps (it’s a scrollable list), or you can skip them if you don’t want
to do that now.

Follow these steps to install the Python extension for VS Code:

- Click on the Extensions icon on the Activity Bar on the left side of the
  Visual Studio Code window (or choose View > Extensions from the menu). The
  icon looks like four squares.
- This will open the Extensions pane on the left side of the window. Type
  “Python” in the search box, then click on “Install” next to the Python
  extension that lists Microsoft as the developer:
- After installing the Python extension, you will see a “Get started with Python
  development” tab and a “Get started with Jupyter Notebooks” tab. You can close
  these.

Follow these steps to install two more extensions that will be useful. These are
optional, but they make it easier to read and edit data stored in text files,
such as the .csv files used by Switch:

- Type “rainbow csv” in the search box in the Extensions pane, then click on
  “Install” next to the Rainbow CSV extension (this is optional, but makes it
  easier to read and edit data stored in text files, such as the .csv files used
  by Switch):
- Type “excel viewer” in the search box, then click to install the Excel Viewer
  extension (this is also optional, but gives a nice grid view of .csv files):

# Install Python/conda environment

We recommend that you install Miniforge for your platform following the
instructions below. If you already have Miniforge, Miniconda or Anaconda, you
can skip ahead to the next section. 

## Windows

Download and run the Miniforge installer from
[https://conda-forge.org/download/]. You may ignore the options to "Add
Miniforge3 to my PATH environment variable" and "Register Miniforge3 as my
default Python".

After installing Miniforge, open VS Code, then choose File > Open Folder... and
choose your home folder, `C:\Users\<yourname>`. Then choose Terminal > New
Terminal, which will open a command prompt in your home folder. Then type these
commands in the terminal pane. These will make the conda environment manager
available within the command line environment:

```
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\miniforge3\condabin\conda init --all
```

Then close the terminal pane by clicking the trash can icon next to the name of
the terminal on the right side (you may need to hover your cursor over the name
to get it to appear).

## Linux/macOS

Miniforge does not have a graphical installer for Linux and macOS. The easiest
way to insall on these platforms is to open VS Code, choose Terminal > New
Terminal, then run the command below. This will download and run the text-based
installer. 

```
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" && bash Miniforge3-$(uname)-$(uname -m).sh -u && rm Miniforge3-$(uname)-$(uname -m).sh
```

When prompted, say `yes` to the option to "update your shell profile to
automatically initialize conda".

After installation finishes, type `exit` to close the VS Code terminal pane.

If the `curl` command above doesn't work, you could instead follow the
installation instructions at [https://conda-forge.org/download/].

On macOS, if you prefer a graphical installer instead of the text-based
installer above, you could instead install Miniconda from
[https://www.anaconda.com/docs/getting-started/miniconda/install]. Licensing for
Miniconda is more restrictive than Miniforge and you will need to create an
Anaconda account and receive commercial emails (at least until you unsubscribe).
Or you could avoid the login and emails by installing a Miniconda .pkg file from
[https://repo.anaconda.com/miniconda/].


# Setup modeling software

Open VS Code if it is not open already. 

Choose File > Open Folder... and open the parent folder where you want to place
this repository. e.g., if you want the model to be stored in
a Switch-USA-PG-ReEDS folder inside Documents, open Documents now.

Open a new terminal pane: Terminal > New Terminal

Run these commands in the terminal pane:

```
# update conda itself
conda update -y -n base -c conda-forge conda

# Create Switch/PowerGenome Python environment (may take a few minutes to solve environment)
conda env create -n switch-pg-reeds --env-spec environment.yml --file https://github.com/switch-model/Switch-USA-PG-ReEDS/raw/refs/heads/main/environment.yml

# Activate the new environment
conda activate switch-pg-reeds
 
# clone the Switch-USA-PG-ReEDS repository and PowerGenome submodule
git clone https://github.com/switch-model/Switch-USA-PG-ReEDS --recurse-submodules --depth=1

# install PowerGenome from the local sub-repository
cd Switch-USA-PG-ReEDS
cd PowerGenome
pip install -e .
```

Next choose File > Open Folder..., navigate to the Switch-USA-PG-ReEDS folder
you just created and choose "Open" or "Select Folder". You can repeat this step
anytime you want to work with this repository in the future. (You may now close 
the window you previously had open on the parent folder.)

Set VS Code to use the switch-pg-reeds conda environment for any future work in
this directory:

- Press shift-ctrl-P (Windows) or shift-command-P (Mac). 
- Choose `Python: Select Interpreter`.
- Select the switch-pg-reeds environment.

If no environments appear when you choose `Python: Select Interpreter`, go to
File > Preferences > Settings (on Windows) or Code > Settings... > Settings (on
macOS), then search for "locator" and change `Python: Locator` from "native" to
"js". Then try the `Python: Select Interpreter` command again.

# Download and patch PowerGenome input data and configure PowerGenome to use it

In VS Code, choose Terminal > New Terminal, then run these commands in the
terminal pane (inside the Switch-USA-PG-ReEDS directory):

```
# next line is not needed if prompt already says (switch-pg-reeds)
conda activate switch-pg-reeds

# download upstream data used by PowerGenome
python download_pg_data.py

# apply expected coal plant retirements to the EIA 860m workbook
python update_coal_closures.py

# create the custom load profiles used for this study 
# (too large to store on github)
python make_study_loads.py
```

If you prefer to use the ReEDS standard loads instead of the our load growth
schedule, you can skip the last command above and instead comment out
`regional_load_fn` in `pg/settings/demand.yml/` and set `load_source_table_name:
load_curves_nrel_reeds` in the same file.

# Notes about PowerGenome scenario configuration

`pg/settings/` holds the settings used for all scenarios in this study in a
collection of `*.yml` files. In addition to these, tabular data is stored in
`*.csv` files. The location of the .csv files and the specific files to use for
the study are identified in `extra_inputs.yml`. The location should be a
subdirectory (currently `extra_inputs`) at the same level as the `settings`
folder that holds the .yml files. One special .csv file, identified by the
`scenario_definitions_fn` setting (currently
`pg/extra_inputs/scenario_inputs.csv`), defines all the cases available and
identifies named groups of settings to use for various aspects of the model for
each one. Each setting defined in the scenario definitions csv turns flags on
and off in the `settings_management` key (in
`pg/settings/scenario_management.yml`). These in turn override specific keys in
all the *.yml files with new values reflecting the flag setting.

Currently, the most useful flag is `time_series`, which can be all 7 years of
data (case ID `p1`), 4 sample weeks (case ID `s4` and `s4_flat`) or 20x1 sample
days (case ID `s20x1`, `s20x1_flat` and `s20x1_decarb`). Cases ending with
`_flat` have no load growth, and cases with `_decarb` have deep decarbonization
($200 national carbon tax).

# Generate Switch inputs

To setup one model case for one year for testing, you can run this command in
the VS Code terminal pane (after running `conda activate switch-pg-reeds` if
needed):

```
# setup one example case (specify case-id and year)

# s20x1 uses our standard assumptions and 20x1 day time sampling;
# use s4x1 if you need a smaller and faster model for testing.
python pg_to_switch.py pg/settings switch/in/ --case-id s20x1 --year 2030

# (p1 uses all weather years (2006-13) as a single timeseries, which can't
# be solved but is relatively quick to generate and useful for inspection)
python pg_to_switch.py pg/settings switch/in/ --case-id p1 --year 2030
```

The `pg_to_switch.py` script uses settings from the first directory you specify
(`pg/settings`) and places Switch model input files below the second directory
you specify (`switch/in/`), e.g., in `switch/in/2030/s20x1/`.

To generate data for a specific model case, use `--case-id <case_name>`. To
generate data for multiple cases, use `--case-id <case_1> --case-id <case_2>`,
etc. If you omit the `--case-id` flag, `pg_to_switch.py` will generate inputs
for all available cases.

Similarly, to generate data for a specific year, use `--year NNNN`, for multiple
years, use `--year MMMM --year NNNN`, etc. If you omit the `--year` flag,
`pg_to_switch.py` will generate inputs for all available years.

By default, `pg_to_switch.py` will generate foresight models for each case-id
when multiple years are requested. In this case, each model will use all
available years of data. If you'd like to make single-period (myopic) models,
you can use the `--myopic` flag.

For the previous MIP project, most cases were setup as myopic models, where one
model was created for each case for each reference year, then they were solved
in sequence, from the first to the last, with extra code to carry construction
plans and retirements forward to later years. That code is available in this
repository if needed (see `scenarios.txt` files generated by `pg_to_switch.py`
and `switch/study_modules/prepare_next_stage.py`).

(Note: for comparison, you can generate GenX inputs by running `mkdir -p
genx/in`, then `run_powergenome_multiple -sf pg/settings -rf genx/in -c s20x1`.
They will stored in `genx/in`.)

# Run Switch

You can solve individual cases like this:

```
cd switch
# 4 x 1-day sample timeseries (good test case)
switch solve --inputs-dir in/2030/s4x1 --outputs-dir out/2030/s4x1
# all-weather-years (not actually solvable)
switch solve --inputs-dir in/2030/p1 --outputs-dir out/2030/p1
```

(If you are using PowerShell in Windows, you will need to type `switch.exe` to
avoid a conflict with its built-in `switch` command. You can get around this by
going to File > Preferences > Settings and changing "Terminal Integrated Default
Profile: Windows" to "Command Prompt".)

(to add: solving batches of cases, customizing scenarios, possibly multi-year
and/or myopic cases, based on
https://github.com/switch-model/Switch-USA-PG/blob/main/README.md)

# Updating local copy of the repository

To update this repository and the PowerGenome submodule with the latest code
from github, use

```
git pull --recurse-submodules
```

# Replicating this repository

The data used in this study fall into two categories:

- Settings and data stored in this github repository itself
  (https://github.com/switch-model/Switch-USA-PG-ReEDS)
- External data downloaded or created by the scripts discussed above

This section provides some information on the origin of these. Users do not need
to recreate these, since they are either part of the repository already,
downloaded, or created automatically by the scripts run above. However, this
information may be useful for people who want to replicate the upstream inputs
or modify this workflow.

## PowerGenome settings files

The settings files (`pg/` directory of this repository) were originally prepared
by Greg Schivley in May 2025 and uploaded to a reeds-ba branch of the
PowerGenome/PowerGenome-examples repository. Matthias Fripp downloaded them to
`pg/`from
https://github.com/PowerGenome/PowerGenome-examples/tree/reeds-ba/ReEDS-BA and
manually added or changed settings in pg/settings/*.yml to define inputs for
this study. Parts of these files are also written automatically by
`make_emission_policies.py`, discussed below.

## External data

The most important category of external data are the standard PowerGenome input
data and resource profiles created by Greg Schivley. These are documented in
pg_data.yml and downloaded by the `download_pg_data.py` script. 

As noted in pg_data.yml, several files in the resource_groups folder had errors
when downloaded from Greg Schivley's Google Drive, so we patched them with
`patch_pg_resource_groups.py` and uploaded the patched folders to Matthias
Fripp's Google Drive for downloading by `download_pg_data.py`. If you would like
to re-create these files directly from Greg Schivley's versions, you can just
run `python patch_pg_resource_groups.py`. That will download data from Greg
Schivley's Google Drive and use it to create
`pg_data/existing_resource_groups-patched/` and `pg_data/ReEDS-cpas-patched/`.
(In this case, you can remove references to these folders from the
`download_gdrive_files` section of `pg_data.yml`.)

Similarly, `pg_data.yml` points to a "retro" version of the PUDL database
(mostly from EIA) that Matthias Fripp created by copying data from the August
2025 edition of PUDL into a new sqlite database with the PUDL schema used before
December 2023. The `make_retro_pudl_data.py` script does this. See that script
for additional information.

After downloading the main data for PowerGenome, scripts were used to create
various categories of additional input data for PowerGenome, as discussed below.
The outputs of these scripts are part of the repository, but the scripts can be
revised and/or re-run if needed to re-create or update the input data. These 
should be run before running pg_to_switch.py to create Switch model inputs.

**Load profiles, load growth forecasts and international exports.** Matthias
Fripp created these using make_study_loads.py (which contains some additional
documentation). The resulting inputs for PowerGenome are in the
`pg/extra_inputs` folder of this repository. The **baseline load profiles** come
from the load_curves_nrel_reeds in pg_data/pg_misc_tables_efs_2025.3.sqlite (see
pg_data.yml) and may originate directly from
[ReEDS](https://github.com/NREL/ReEDS-2.0/tree/main/inputs/load). **Zonal load
growth rates** come from an [ICF
study](https://www.icf.com/insights/energy/demand-growth-challenges-opportunities-utilities)
and were converted from map images to zonal data via
`growth_rates/retrieve_icf_growth.py`, including manual image editing at several
stages documented there (mainly to remove text that obscured map colors and
ensure that borders were correct). **International exports** come from EIA
interchange data via PUDL. This model treats them as additional loads in the
exporting zone.

**Clean energy standards (CES), Renewable Portfolio Standards (RPS),**
**minimum/maximum capacity requirements and carbon policies.** The
`make_emission_policies.py` script implements these by reading rules from the
ReEDS repository on github.com and creating several files and .yml sections with
equivalent terms in the pg/settings and pg/extra_inputs directories. This also
creates a national cap on new wind development equal to the highest rate of
growth in 2014-24, as determined from EIA 860m data downloaded by PowerGenome.
These are the files and sections created by this script:

  - `emission_policies_current.csv`: current CO2 policies
  - `emission_policies_decarb.csv`: national cap / carbon tax case
  - `model_definition.yml/generator_columns` and `resource_tags.yml/model_tag_names`: RPS, CES and minimum-capacity tags that should be attached to generators (just defines the tags, doesn't assign values)
  - `scenario_management.yml/settings_management/[various years]/all_cases`: levels for minimum and maximum capacity requirements
  - `regional_resource_tags.yml`: generator eligibility for state RPS, CES and minimum-capacity programs
    - eligibility for max-capacity programs (the national wind ban (`MaxCapTag_WindGrowth`) and an optional ban on individual technologies (`MaxCapTag_Ban`)) are specified manually in `model_definition.yml/generator_columns`, `resource_tags.yml/model_tag_values` and possibly `scenario_management.yml/settings_management/` as needed.

**Coal plant closures**. The `update_coal_closures.py` script retrieves data on
expected near-term coal plant closures from [Global Energy
Monitor](https://globalenergymonitor.org/projects/global-coal-plant-tracker/download-data/)
and uses them to update retirement dates in the local copy of EIA 860m data
maintained by PowerGenome.

# Generating Switch inputs on a high performance computing (HPC) cluster

On an HPC system that uses the slurm scheduling manager, the cases can be setup
in parallel as follows:

```
# generate all the cases (up to 24 unique ones) defined in scenario_inputs.csv
sbatch pg_to_switch.slurm pg/settings switch/in/ --myopic

# generate two specific cases (--array argument tells which items from the 
# --case-id list to run)
sbatch --array=1-2 pg_to_switch.slurm pg/settings switch/in/ --case-id p1 --case-id s20x1 --myopic
```

The `--myopic` flag is used here to ensure that the 2024 and 2030 cases 
are setup as separate models.

The `pg_to_switch.slurm` batch definition will run multiple copies of the
`pg_to_switch.py` script in an array with the arguments provided. It passes the
task ID of each job within the array (by default elements 1-24) as a
`--case-index` argument to the `pg_to_switch.py` script, which causes
`pg_to_switch.py` to just setup that one case number from among all cases
identified on the command line. You can adjust the `--array=n-m` at the start
and `--case-id` arguments later in the line to choose which cases to prepare,
e.g. this will just setup `s20x1`:

```
sbatch --array=1 pg_to_switch.slurm pg/settings switch/in/ --case-id s20x1 --myopic
```

As an alternative, you can run `sbatch setup_cases.slurm`, which will run
`setup_cases.sh`. This will prepare all the cases one by one using a single
machine.
