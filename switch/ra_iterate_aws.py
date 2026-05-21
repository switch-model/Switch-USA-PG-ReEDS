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

# TODO: cap the number of tasks in the array at the number of
# scenarios in the specified list (for solve-scenarios cases)
# to avoid over-provisioning on the later CE runs (not much of
# a problem if using fargate and/or computing environment is
# limited to instances that exactly fit one worker)

# def prune_empty(spec):
#     """
#     Remove empty leaves and branches from the spec to avoid errors from
#     empty overrides when submitting the job. Convenience function so
#     the job specs can just be copied and pasted from AWS
#     """
#     if isinstance(spec, dict):
#         for k, v in spec.items():
#             if isinstance(v, (dict, list)):
#                 prune_empty(v)
#                 if not v:
#                     del spec[k]
#     elif isinstance(spec, list):
#         for i in range(len(spec)-1, 0):
#             if isinstance(v, (dict, list)):
#                 prune_empty(spec[i])
#                 if not spec[i]:
#                     del spec[i]


def submit_job(batch, jobs, job_name, dep_id=None):
    spec = jobs[job_name]
    if dep_id is not None:
        spec["dependsOn"] = [{"jobId": dep_id}]
    print("submitting job:")
    print(spec)
    resp = batch.submit_job(**spec)
    job_id = resp["jobId"]
    dep = f", dependent on {dep_id}" if dep_id else ""
    print(f"Submitted job {job_name}: {job_id}{dep}")
    return job_id


def main():
    with open("ra_jobs.json") as f:
        jobs = json.load(f)

    # clear out the default scenario queue so new ones without override will run
    if os.path.exists("scenario_queue"):
        shutil.rmtree("scenario_queue")

    batch = boto3.client("batch", region_name=jobs["region"])

    ce_scens_file = jobs["ce_scenario_list"]
    with open(ce_scens_file, "r") as f:
        # any scenarios to run?
        done = not f.read().strip()

    if done:
        print(f"Iteration is complete: no more scenarios to run in {ce_scens_file}.")
        print("Solving evaluation scenarios.")
        submit_job(batch, jobs, "solve_ce_eval")
        return

    # still have some RA scenarios to run; setup another iteration
    # run these jobs as a chain (the last one calls this script again)
    job_queue = [
        "solve_ce",
        "solve_ra",
        "ra_add_difficult_timeseries",
        "ra_iterate_aws",
    ]

    prev_id = None
    for job_name in job_queue:
        prev_id = submit_job(batch, jobs, job_name, prev_id)


if __name__ == "__main__" and "ipykernel" not in sys.argv[0]:
    # running as a script, not from a jupyter environment
    main()
