#!/usr/bin/env python3
"""
if anything in build-scens list:
    remove scenario_queue
    submit build-scens job
    submit RA job dependent on build-scens job
    submit ra_add_difficult_timeseries job dependent on RA job
    submit iterate job dependent on ra_add_difficult job
first run: either launch this script's job def or run it directly
"""
import json, shutil, sys, os

import boto3


def main():
    with open("ra_jobs.json") as f:
        jobs = json.load(f)

    ce_scens_file = jobs["ce_scenario_list"]
    with open(ce_scens_file, "r") as f:
        # any scenarios to run?
        done = not f.read().strip()

    if done:
        print(f"Ending iteration; no more scenarios to run in {ce_scens_file}.")
        return

    # have some scenarios to run; setup another iteration

    # clear out the scenario queue so new ones will run
    if os.path.exists("scenario_queue"):
        shutil.rmtree("scenario_queue")

    # setup these jobs as a chain (the last one calls this script again)
    job_queue = [
        "solve_ce",
        "solve_ra",
        "ra_add_difficult_timeseries",
        "ra_iterate_aws",
    ]
    batch = boto3.client("batch", region_name=jobs["region"])

    prev_id = None
    for job_name in job_queue:
        spec = jobs[job_name]
        if prev_id is None:
            # first job
            spec["dependsOn"] = []
        else:
            # each job depends on prior
            spec["dependsOn"] = [{"jobId": prev_id, "type": "SEQUENTIAL"}]
        # print(spec)
        # job_id = "dummy"
        resp = batch.submit_job(**spec)
        job_id = resp["jobId"]
        dep = f", dependent on {prev_id}" if prev_id else ""
        print(f"Submitted job {job_name}: {job_id}{dep}")
        prev_id = job_id


if __name__ == "__main__" and "ipykernel" not in sys.argv[0]:
    # running as a script, not from a jupyter environment
    main()
