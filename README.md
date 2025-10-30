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
[https://conda-forge.org/download/]. We recommend selecting the option to "Add
Miniforge3 to my PATH environment variable" despite the warning. Otherwise
conda and Python won't be able to be found from the VS Code terminal pane later.

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
this repository. e.g., if you open Documents, the steps below will create a
folder for the study called Switch-USA-PG-ReEDS inside the Documents folder.

Open a new terminal pane: Terminal > New Terminal

Run these commands in the terminal pane:

```
# update conda itself
conda update -y -n base -c conda-forge conda

# Create Switch/PowerGenome Python environment (may take a few minutes to solve environment)
# On Windows this will give a file error but still work correctly
conda env create -n switch-pg-reeds --env-spec environment.yml --file https://github.com/switch-model/Switch-USA-PG-ReEDS/raw/refs/heads/main/environment.yml

# Activate the new environment
conda activate switch-pg-reeds
 
# clone the Switch-USA-PG-ReEDS repository and PowerGenome submodule
git clone https://github.com/switch-model/Switch-USA-PG-ReEDS --recurse-submodules --depth=1
cd Switch-USA-PG-ReEDS

# install PowerGenome from the local sub-repository
cd PowerGenome
pip install -e .
cd ..
```

Close the current VS Code window. Then choose File > Open, navigate to the
Switch-USA-PG-ReEDS folder you just created and choose "Open". You can repeat
this step anytime you want to work with this repository in the future.

Set VS Code to use the switch-pg-reeds conda environment for future work: 
- Press shift-ctrl-P (Windows) or shift-command-P (Mac). 
- Choose `Python: Select Interpreter`.
- Select the switch-pg-reeds environment.

# Download and patch PowerGenome input data and configure PowerGenome to use it

In VS Code, choose Terminal > New Terminal, then run these commands in the
terminal pane (inside the Switch-USA-PG-ReEDS directory):

```
conda activate switch-pg-reeds

python download_pg_data.py
python patch_pg_existing_resource_groups.py
```

After this, manually update several .csv data files in `pg_data` as noted in the
output from the `patch_pg_existing_resource_groups.py` script. This may be
easiest to do with a spreadsheet program. (In the future we may post the patched
versions of the input files on our own Google Drive and update pg_data.yml to
download those directly.)

In the VS Code terminal pane, run the command below to create the custom load profiles used for this study (too large to store on github):

```
python make_study_loads.py
```

If you prefer to use the ReEDS standard loads instead, you can skip this step
and instead comment out `regional_load_fn` in `pg/settings/demand.yml/` and set
`load_source_table_name: load_curves_nrel_reeds` in the same file.

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
days (case ID `s20_1`, `s20_1_flat` and `s20_1_decarb`). Cases ending with
`_flat` have no load growth, and cases with `_decarb` have deep decarbonization
($200 national carbon tax).

# Generate Switch inputs

To setup one model case for one year for testing, you can run this command in the VS Code terminal pane (after running `conda activate switch-pg-reeds` if needed):

```
# setup one example case (specify case-id and year)
# (s20_1 uses our standard assumptions and 20x1 day time sampling)
python pg_to_switch.py pg/settings switch/in/ --case-id s20_1 --year 2030
```

The `pg_to_switch.py` script uses settings from the first directory you specify
(`pg/settings`) and places Switch model input files below the second directory
you specify (`switch/in/`).

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
and `switch/mip_modules/prepare_next_stage.py`).

(Note: for comparison, you can generate GenX inputs by running `mkdir -p
genx/in`, then `run_powergenome_multiple -sf pg/settings -rf genx/in -c s20_1`.
They will stored in `genx/in`.)

## Generate Switch inputs on high performance computing (HPC) cluster

On an HPC system that uses the slurm scheduling manager, the cases can be setup
in parallel as follows:

```
sbatch pg_to_switch.slurm MIP_results_comparison/case_settings/26-zone/settings-atb2023 switch/26-zone/in/ --myopic
sbatch --array=1-2 pg_to_switch.slurm MIP_results_comparison/case_settings/26-zone/settings-atb2023 switch/26-zone/in/ --case-id base_20_week --case-id current_policies_20_week
```

The `pg_to_switch.slurm` batch definition will run multiple copies of the
`pg_to_switch.py` script in an array with the arguments provided. It passes the
task ID of each job within the array (by default elements 1-24) as a
`--case-index` argument to the `pg_to_switch.py` script, which causes
`pg_to_switch.py` to just setup that one case number from among all cases
identified on the command line. You can adjust the `--array=n-m` at the start
and `--case-id` arguments later in the line to choose which cases to prepare,
e.g. this will just setup `base_short_retire`:

```
sbatch --array=1 pg_to_switch.slurm MIP_results_comparison/case_settings/26-zone/settings-atb2023 switch/26-zone/in/ --myopic --case-id base_short_retire
```

As an alternative, you can run `sbatch setup_cases.slurm`, which will run
`setup_cases.sh`. This will prepare all the cases one by one using a single
machine.

## Generate inputs from ReEDS data

First, put these entries in `pg_data.yml`:

```
  pg_data/PowerGenome Data Files/pg_misc_tables_efs_2025.3.sqlite.zip: https://drive.google.com/file/d/1TR-bQ0vnE3pgNsl0opk03PFiPBkGIMB3/view?usp=sharing
  ...
  PG_DB: pg_data/PowerGenome Data Files/pg_misc_tables_efs_2025.3.sqlite  # ReEDS compatible, May 2025
```

Then run `python download_pg_data.py misc_tables`.

Then get ReEDS-BA settings:

```
git clone --branch reeds-ba --single-branch https://github.com/PowerGenome/PowerGenome-examples.git
```

Generate model inputs:

```
# sampled case, not currently working; see options in 
# PowerGenome-examples/ReEDS-BA/extra_inputs/scenario_inputs.csv
python pg_to_switch.py PowerGenome-examples/ReEDS-BA/settings switch/reeds/in/ --case-id s1

# all-hours case (7 years!)
python pg_to_switch.py PowerGenome-examples/ReEDS-BA/settings switch/reeds/in/ --case-id p1
```

# Run Switch

You can solve one case for one year like this:

```
cd switch
switch solve --inputs-dir in/2030/p1 --outputs-dir out/2030/p1
```

This works well for foresight cases or single-period cases, which only have one
model to solve per case. However, for the myopic cases, it is necessary to solve
each year in turn and chain the results forward to the next stage. The chaining
can be done by adding `--include-module mip_modules.prepare_next_stage` to the
command line for all but the last stage and adding `--input-aliases
gen_build_predetermined.csv=gen_build_predetermined.chained.base_short.csv
gen_build_costs.csv=gen_build_costs.chained.base_short.csv
transmission_lines.csv=transmission_lines.chained.base_short.csv` for all but
the first stage. (The `prepare_next_stage` module prepares alternative inputs
for the next stage that include the construction plan from the current stage.
Then the `--input-aliases` flag tells Switch to use those alternative inputs.)

So you _could_ solve the myopic version of the `base_short` model with these
commands (but there's a better option, see below):

```
cd switch
switch solve --inputs-dir 26-zone/in/2027/base_short --outputs-dir 26-zone/out/2027/base_short  --include-module mip_modules.prepare_next_stage
switch solve --inputs-dir 26-zone/in/2030/base_short --outputs-dir 26-zone/out/2030/base_short  --include-module mip_modules.prepare_next_stage --input-aliases gen_build_predetermined.csv=gen_build_predetermined.chained.base_short.csv gen_build_costs.csv=gen_build_costs.chained.base_short.csv transmission_lines.csv=transmission_lines.chained.base_short.csv
switch solve --inputs-dir 26-zone/in/2035/base_short --outputs-dir 26-zone/out/2035/base_short  --include-module mip_modules.prepare_next_stage --input-aliases gen_build_predetermined.csv=gen_build_predetermined.chained.base_short.csv gen_build_costs.csv=gen_build_costs.chained.base_short.csv transmission_lines.csv=transmission_lines.chained.base_short.csv
switch solve --inputs-dir 26-zone/in/2040/base_short --outputs-dir 26-zone/out/2040/base_short  --include-module mip_modules.prepare_next_stage --input-aliases gen_build_predetermined.csv=gen_build_predetermined.chained.base_short.csv gen_build_costs.csv=gen_build_costs.chained.base_short.csv transmission_lines.csv=transmission_lines.chained.base_short.csv
switch solve --inputs-dir 26-zone/in/2045/base_short --outputs-dir 26-zone/out/2045/base_short  --include-module mip_modules.prepare_next_stage --input-aliases gen_build_predetermined.csv=gen_build_predetermined.chained.base_short.csv gen_build_costs.csv=gen_build_costs.chained.base_short.csv transmission_lines.csv=transmission_lines.chained.base_short.csv
switch solve --inputs-dir 26-zone/in/2050/base_short --outputs-dir 26-zone/out/2050/base_short  --input-aliases gen_build_predetermined.csv=gen_build_predetermined.chained.base_short.csv gen_build_costs.csv=gen_build_costs.chained.base_short.csv transmission_lines.csv=transmission_lines.chained.base_short.csv
```

To simplify solving myopic models, `pg_to_switch.py` creates scenario definition
files in the `switch/26-zone/in` directory, with names like
`scenarios_<case_name>.txt`. The `switch solve-scenarios` command can use these
to solve all the steps in sequence. (Each one contains the command line flags
needed for each stage of the model, and `swtich solve-scenarios` solves each one
in turn.) So you can solve the reference case (`base_52_week`) with this
command:

```
cd switch
switch solve-scenarios --scenario-list 26-zone/in/scenarios_base_52_foresight.txt
```

The `pg_to_switch.py` command also creates scenario definition files for some
alternative cases that share the same inputs directory as the standard cases,
but use alternative versions of some input files (currently only the carbon
price file). The definitions for these can also be found in `26-zone/in/`, and
they can be solved the same way as the standard cases, e.g.,
`switch solve-scenarios --scenario-list 26-zone/in/scenarios_base_52_week_co2_50.txt`.
You can also look inside these to see the extra flags used setup these cases.

To run all the cases for the MIP study, you can use the following commands:

```
cd switch

# myopic cases
switch solve-scenarios --scenario-list 26-zone/in/scenarios_base_20_week.txt
switch solve-scenarios --scenario-list 26-zone/in/scenarios_base_52_week.txt
switch solve-scenarios --scenario-list 26-zone/in/scenarios_base_52_week_co2_50.txt
switch solve-scenarios --scenario-list 26-zone/in/scenarios_base_52_week_co2_1000.txt
switch solve-scenarios --scenario-list 26-zone/in/scenarios_base_52_week_commit.txt
switch solve-scenarios --scenario-list 26-zone/in/scenarios_base_52_week_no_ccs.txt
switch solve-scenarios --scenario-list 26-zone/in/scenarios_base_52_week_retire.txt
switch solve-scenarios --scenario-list 26-zone/in/scenarios_base_52_week_tx_0.txt
switch solve-scenarios --scenario-list 26-zone/in/scenarios_base_52_week_tx_15.txt
switch solve-scenarios --scenario-list 26-zone/in/scenarios_base_52_week_tx_50.txt
switch solve-scenarios --scenario-list 26-zone/in/scenarios_current_policies_20_week.txt
switch solve-scenarios --scenario-list 26-zone/in/scenarios_current_policies_52_week.txt
switch solve-scenarios --scenario-list 26-zone/in/scenarios_current_policies_52_week_commit.txt
switch solve-scenarios --scenario-list 26-zone/in/scenarios_current_policies_52_week_retire.txt

# foresight cases
switch solve-scenarios --scenario-list 26-zone/in/scenarios_base_20_week_foresight.txt
switch solve-scenarios --scenario-list 26-zone/in/scenarios_current_policies_20_week_foresight.txt
```

Note: If you ever need to manually create next-stage inputs from a previous
stage's outputs, you can run a command like this:

```
cd switch
# prepare 2035 inputs from 2030 model (specify 2030 inputs and outputs directories)
python -m mip_modules.prepare_next_stage 26-zone/in/2030/base_short 26-zone/out/2030/base_short
# or:
python mip_modules/prepare_next_stage.py 26-zone/in/2030/base_short 26-zone/out/2030/base_short
```

# Prepare result summaries for comparison

After solving the models, run these commands to prepare standardized results and
copy them to the `MIP_results_comparison` sub-repository.

```
cd MIP_results_comparison
git pull
cd ../switch
python save_mip_results.py
cd ../MIP_results_comparison
git add .
git commit -m 'new Switch results'
git push
```

TODO: maybe move all of this into a switch module so it runs automatically when
each case finishes

# Updating repository with upstream changes

To update this repository and all the submodules (PowerGenome and
MIP_results_comparison), use

```
git pull --recurse-submodules
```

To update a submodule, `cd` into the relevant directory and run `git pull`. Then
run `git add <submodule_dir>` and `git commit` in the main Switch-USA-PG
directory to save the updated submodules in the Switch-USA-PG repository. This
will save pointers in Switch-USA-PG showing which commit we are using in each
submodule.

# Replicating this repository

The data used in this study fall into two categories:

- Settings and data stored in this github repository (https://github.com/switch-model/Switch-USA-PG-ReEDS)
- External data downloaded or created by the scripts discussed above

This section provides some information on the origin of these. Users do not need
to recreate these, since they are either part of the repository already,
downloaded, or created automatically by the scripts run above. However, this
information may be useful for people who want to replicate the upstream inputs
or modify this workflow.

## PowerGenome settings files

The settings files (`pg/` directory of this repository) were originally prepared
by Greg Schivley in May 2025 and uploaded to a reeds-ba branch of the
PowerGenome/PowerGenome-examples repository. Matthias Fripp downloaded them from
https://github.com/PowerGenome/PowerGenome-examples/tree/reeds-ba/ReEDS-BA and
manually added settings to pg/settings/*.yml to customize it for this study. It
is probably best to think of the *.yml files as being custom-written to define
the inputs that are wanted for this particular study. Parts of these are also
written automatically by `make_emission_policies.py`, discussed below.

## External data

The most important category of external data are the standard PowerGenome inputs
and resource profiles created by Greg Schivley. These are documented in
pg_data.yml and downloaded by the `download_pg_data.py` script. Upon download,
these files have some errors, which are patched by
`patch_pg_existing_resource_groups.py` or manually by users, based on notes
printed from that script.

In addition to the inputs from Greg Schivley, pg_data.yml points to a "retro"
version of the PUDL database (mostly from EIA) that Matthias Fripp created by
copying data from the August 2025 edition of PUDL into a new sqlite database
with the PUDL schema used before December 2023. The `make_retro_pudl_data.py`
script does this. See that script for additional information.

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

**Clean energy standards (CES), Renewable Portfolio Standards (RPS),
minimum/maximum capacity requirements and carbon policies.** The
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
