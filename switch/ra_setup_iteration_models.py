"""
Setup capacity expansion (CE) and resource adequacy models for iteration.

This script does the following:

- Copy CE model inputs from the directories specified in `ce_scens_file` to
  `ra/scenario_name` subdirectories. The new copies can then be updated with
  additional timeseries by `ra_add_difficult_timeseries.py`.
- Create new versions of `ce_scens_file` and `ra_scens_file` with ".ra" as an
  additional suffix. These can then be passed as arguments to
  `ra_add_difficult_timeseries.py`, which will use them to identify CE and RA
  scenarios to run, and update them at each iteration with definitions of the
  models to be run at the next iteration.
"""

import argparse, shlex, shutil, sys, os
from pathlib import Path

from ra_add_difficult_timeseries import get_scenario_args


def get_script_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "ce_scens_file",
        type=str,
        help="""
            Scenario list file used to define capacity expansion scenarios,
            e.g., in/2030/s20x1/scenarios_build.txt (usually created by
            adjust/define_scenarios.py). This script will create a new version
            for resource adequacy iteration with extension .ra.txt.
        """,
    )
    parser.add_argument(
        "ra_scens_file",
        type=str,
        help="""
            Scenario list file used to define resource adequacy (split)
            scenarios, e.g., in/2030/resource_adequacy/scenarios_split.txt
            (usually created by adjust/create_split_models.py). This script will
            create a new version for resource adequacy iteration with extension
            .ra.txt.
        """,
    )
    options = parser.parse_args()
    return options


# testing:
"""
options = lambda: None # dummy namespace object
options.ce_scens_file = "in/2030/s20x1/scenarios_build.txt"
options.ra_scens_file = "in/2030/resource_adequacy/scenarios_split.txt"
"""


def main(options):
    # simple version: everything happens in the same outputs directory as the original
    # model, so we just
    # - copy files from in/s20x1 to in/s20x1/ra/scenario_name
    # - copy scenarios_build.txt to scenarios_build_ra.txt with different input dirs
    # - copy scenarios_split.txt to scenarios_split_ra.txt
    # fancy version: like above, but we change the outputs dir for the CE models and the
    # reuse-dir for the RA models.

    def ce_ra_path(ce_scen):
        return Path(ce_scens[ce_scen]["inputs_dir"]) / "ra" / ce_scen

    def make_ce_ra_dir(ce_scen):
        """
        Create ce_dir/ra/ce_scen_name with a copy of all files (but not
        subdirs) from ce_dir.
        """
        ce_dir = Path(ce_scens[ce_scen]["inputs_dir"])
        ce_ra_dir = ce_ra_path(ce_scen)
        if ce_ra_dir.is_dir():
            shutil.rmtree(ce_ra_dir)
        ce_ra_dir.mkdir(parents=True)
        for p in ce_dir.iterdir():
            if p.is_file():
                shutil.copy2(p, ce_ra_dir / p.name)
        print(f"Created CE RA model directory `{ce_ra_dir}`.")

    if options.ce_scens_file.endswith(".ra.txt") or options.ra_scens_file.endswith(
        ".ra.txt"
    ):
        print(
            "WARNING: the specified scenario list files already have .ra.txt "
            "suffix(es); this script should be called with the original scenario "
            "list files, and then it will add the .ra.txt suffixes."
        )

    ce_scens = get_scenario_args(options.ce_scens_file)
    ra_scens = get_scenario_args(options.ra_scens_file)

    # create separate input directories for each CE scenario so they can
    # have different timeseries added; update input dirs to match
    for ce_scen, info in ce_scens.items():
        make_ce_ra_dir(ce_scen)
        info["inputs_dir"] = str(ce_ra_path(ce_scen))

    # generate scenario list files for the RA iteration, using the new
    # locations. These will have the original names plus a .ra.txt suffix.
    new_files = []
    for scens_file, scens in [
        (options.ce_scens_file, ce_scens),
        (options.ra_scens_file, ra_scens),
    ]:
        args = [
            [
                f"--scenario-name",
                info["scenario_name"],
                f"--inputs-dir",
                info["inputs_dir"],
                f"--outputs-dir",
                info["outputs_dir"],
            ]
            + info["args"]
            for info in scens.values()
        ]
        # create a .ra.txt version of the scenario list file
        p = Path(scens_file)
        new_file = str(p.with_suffix(".ra" + p.suffix))
        new_files.append(new_file)
        with open(new_file, "w") as f:
            f.writelines(shlex.join(a) + "\n" for a in args)
        print(f"Created scenario list `{new_file}`.")

    print("\nCreated RA model folders and scenario definitions.")
    print("Next, run the following commands:\n")
    for scens_file in new_files:
        print(f"switch solve-scenarios --scenario-list {scens_file}")
    print(f"python ra_add_difficult_timeseries.py {' '.join(new_files)}")


if __name__ == "__main__" and "ipykernel" not in sys.argv[0]:
    # running as a script, not from a jupyter environment
    main(get_script_args())
