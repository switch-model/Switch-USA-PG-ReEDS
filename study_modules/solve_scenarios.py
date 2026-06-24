#!/usr/bin/env python
# Copyright (c) 2015-2024 The Switch Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0, which is in the LICENSE file.

"""Scenario management module.
Reads scenario-related arguments from the command line and the same options
file that solve.py would read, and uses them to setup scenarios_to_run().
For each scenario, this generator yields a tokenized list of arguments that
define that scenario (similar to sys.argv, but based on a line from a scenario
definition file, followed by any options specified on the command line).
Then it calls solve.main() with this list of arguments (once for each scenario).

A queueing system (based on lock directories within a queue directory) is used to
ensure that scenarios_to_run() will always return the next unsolved
scenario from the scenario list file, even if the file is edited while this
script is running. This makes it possible to amend the scenario list while
long solver jobs are running. Multiple solver scripts can also use
scenarios_to_run() in separate processes to select the next job to run.
"""

import sys, os, random, time
import argparse, shlex, socket, io, glob, multiprocessing
from collections import OrderedDict

from switch_model.utilities import _ArgumentParser

# load the solve module from the same package as this module
from switch_model import solve

# retrieve base options and command-line arguments
option_file_args = solve.get_option_file_args()
cmd_line_args = sys.argv[1:]

# Parse scenario-manager-related command-line arguments.
# Other command-line arguments will be passed through to solve.py via
# scenario_cmd_line_args
parser = _ArgumentParser(
    allow_abbrev=False, description="Solve one or more Switch scenarios."
)
parser.add_argument(
    "--scenario",
    "--scenarios",
    nargs="+",
    dest="scenarios",
    default=[],
    action="extend",
)
parser.add_argument("--scenario-list", default="scenarios.txt")
parser.add_argument("--scenario-queue", default="scenario_queue")
parser.add_argument("--job-id", default=None)

# import pdb; pdb.set_trace()
# get a namespace object with successfully parsed scenario manager arguments
scenario_manager_args = parser.parse_known_args(args=option_file_args + cmd_line_args)[
    0
]
# get lists of other arguments to pass through to standard solve routine
scenario_option_file_args = parser.parse_known_args(args=option_file_args)[1]
scenario_cmd_line_args = parser.parse_known_args(args=cmd_line_args)[1]

# create a logger for this module's output based on default arguments
logger = solve.make_logger(
    solve.parse_pre_module_options(scenario_option_file_args + scenario_cmd_line_args)
)

requested_scenarios = scenario_manager_args.scenarios
scenario_list_file = scenario_manager_args.scenario_list
scenario_queue_dir = scenario_manager_args.scenario_queue

# Get a unique task id.
# This is used to requeue any scenario that this task was working on that got
# interrupted. This is useful for running jobs on a pre-emptable cluster.
# Note: in the past we have tried to get a persistent ID for each parallel task
# by inspecting the cluster computing batch environment or looking at the parent's
# pid (useful when launching several instances of `switch solve-scenarios` in
# different terminals on a desktop). However, that only works if tasks are
# restarted under similar conditions. It also fails if users run one job on a
# cluster that launches several instances of solve-scenarios via a direct call to
# "srun" or "mpirun". That launches many tasks that all end up thinking they're
# the same task and race to reset the queue. So now it is up to the user to
# specify a unique task id in an environment variable or command-line argument.
# If a job id is not specified, interrupted jobs will not be restarted.
job_id = scenario_manager_args.job_id
if job_id is None:
    job_id = os.environ.get("SWITCH_JOB_ID")
if job_id is None:
    # this cannot be running in parallel with another task with the same pid on
    # the same host, so it's safe to requeue any jobs with this id
    job_id = socket.gethostname() + "_" + str(os.getpid())

# TODO: other options for requeueing jobs:
# - use file locks on lockfiles: lock a
# lockfile in the scenario queue corresponding to the name of the scenario, then
# run the scenario, then create
# a flag file indicating the scenario has been run, then unlock the scenario.
# (or maybe just update a "completed.txt" file when it's finished, and use locks
# on that to manage contention; but still use a lockfile to identify jobs that
# are currently running).
# This allows robust kill/restart of the scenario solver without needing task IDs,
# and without creating all the scenario subdirs (so scenario queue can be cleaned
# out more easily too).
# The lockfile package might be able to do this (seems to work OK on the UH HPC,
# which uses lustre shared file system), but it uses flock on Linux, which is not
# supposed to work across NFS. A better option could be to use fcntl.lockf();
# see here for info about unix file locking:
# http://chris.improbable.org/2010/12/16/everything-you-never-wanted-to-know-about-file-locking/
# ***
# A more robust and platform independent way might be to write a hostname,
# port and key (e.g., creation time) into a lockfile; then listen on that port
# (probably with another subprocess). When another machine wants to run the same
# scenario, it first connects to that host and port and offers the key; if
# if the listening process exists and has ever posted that key, it responds
# affirmatively (that the job is either running or [recently] finished running
# on that host), and the attempter moves on. Otherwise it replaces that file
# with a new one (can that be done atomically? might have to do this with lock dirs).
# This requires a way
# for each host to identify itself in an externally accessible way; we could
# just use the IP address and require all hosts to be on the same subnet.
# There's a little more info here about lockfiles:
# http://dev-random.net/linux-lockfile-explained-how-to-use-them-the-easy-or-hard-way/
# ***
# Or maybe it's better just to use a job server approach to manage the available
# scenarios, similar to pyro, or even use pyro or mpirun to run parallel jobs
# directly.
# ***
# Or we could use a sqlite database and just require each worker to update the
# fact that it's still running a particular job, at least once per minute (probably
# via a separate subprocess). If another worker finds a job that was last updated
# more than a minute ago, it can take over and solve it. Race conditions would be
# managed via normal database locks.
# update locktable set host=myhost, time=mytime where host=oldhost and time=oldtime"
# then check whether that was successful, or possibly got updated first by a different
# worker, then launch the job. Or do the check and update within a transaction, so
# it's atomic.
# This makes it harder to selectively restart jobs, but that could be resolved by
# creating "scenario_done" files for each finished scenario (and removing it from
# the DB), so users can restart scenarios by deleting the 'done' file.
# But this requires synchronized clocks across workers...

running_scenarios_file = os.path.join(scenario_queue_dir, job_id + "_running.txt")

# list of scenarios currently being run by this job (always just one with the current code)
running_scenarios = []

# import pdb; pdb.set_trace()


def main(args=None):
    # make sure the scenario_queue_dir exists (marginally better to do this once
    # rather than every time we need to write a file there)
    try:
        os.makedirs(scenario_queue_dir)
    except OSError:
        pass  # directory probably exists already

    # remove lock directories for any scenarios that were
    # previously being solved by this job but were interrupted
    unlock_running_scenarios()

    for scenario_name, args in scenarios_to_run():
        logger.warning(  # not strictly a warning, but often nice to see in the log
            "\n\n=======================================================================\n"
            + "running scenario {s}\n".format(s=scenario_name)
            + "arguments: {}\n".format(args)
            + "=======================================================================\n"
        )

        # call the standard solve module with the arguments for this particular scenario
        # We run this in its own process to avoid sharing module state info between
        # model instances (e.g., a logger created in Pyomo may grab the current sys.stdout
        # while Switch has temporarily replaced it with a timing counter stream, then
        # keep using that for subsequent instances)
        process = multiprocessing.Process(target=run_scenario, args=(args,))
        process.start()
        process.join()

        # other options:
        # solve.main(args)
        # or
        # subprocess.call(shlex.split("python -m solve") + args) <- omit args from options.txt
        # it should also be possible to use a solver server, but that's not really needed
        # since this script has built-in queue management.

        mark_completed(scenario_name)


def run_scenario(args):
    # reactivate stdin in subprocess
    # from https://stackoverflow.com/questions/30134297/python-multiprocessing-stdin-input
    # also see refs to stdin in https://docs.python.org/3/library/multiprocessing.html
    sys.stdin = os.fdopen(0)
    try:
        solve.main(args)
    except:
        # code run in a subprocess never has an uncaught exception, so the
        # excepthook never gets run, which makes it impossible to use the
        # --debug flag. So we call the excepthook (possibly set by solve.main)
        # directly.
        sys.excepthook(*sys.exc_info())


def scenarios_to_run():
    """
    Generator function which returns argument lists for each scenario that should be run.

    Note: each time a new scenario is required, this re-reads the scenario_list file
    and then returns the first scenario that hasn't already started running.
    This allows multiple copies of the script to be run and allocate scenarios among
    themselves.
    """

    skipped = []
    ran = []

    if requested_scenarios:
        # user requested one or more scenarios
        # just run them in the order specified, with no queue-management
        for scenario_name in requested_scenarios:
            completed = False
            scenario_args = (
                scenario_option_file_args
                + read_scenario_list()[scenario_name]
                + scenario_cmd_line_args
            )
            # flag the scenario as being run; then run it whether or not it was previously run
            checkout(scenario_name, force=True)
            yield (scenario_name, scenario_args)
        # no more scenarios to run
        return

    # no specific scenarios requested
    # Run every scenario in the list, with queue management
    # This is done by repeatedly scanning the scenario list and choosing
    # the first scenario that hasn't been run. This way, users can edit the
    # list and this script will adapt to the changes as soon as it finishes
    # the current scenario.
    first_pass = True
    # there may be hundreds of workers, so we stagger out their starts a bit
    time.sleep(random.random() * 5)
    while True:
        # Identify scenarios that have been run by retrieving the names of the
        # lock directories. This is faster than trying to checkout every one,
        # especially if there are many workers. (This may miss a few that are
        # currently being checked out by other workers, but we'll catch those
        # during the checkouts and retries later.)
        logger.info("Reading scenario definitions.")
        all_scens = read_scenario_list()
        logger.info(
            f"Scanning scenario queue for checked-out scenarios ({scenario_queue_dir})."
        )
        already_run = set(os.listdir(scenario_queue_dir))

        # Identify scenarios that still need to be run
        scen_info = {s: a for s, a in all_scens.items() if s not in already_run}
        del all_scens, already_run

        if not scen_info:
            # no more scenarios to run
            if first_pass:
                cmd = "rmdir /s /q" if os.name == "nt" else "rm -rf"
                logger.warning(
                    "Skipping all scenarios because they have already been solved. "
                    "To run all the scenarios again, remove the scenario queue directory "
                    f"{scenario_queue_dir} or its contents ({cmd} {scenario_queue_dir}) "
                    "and then use switch solve-scenarios again."
                )
            else:
                logger.info("All scenarios have been solved; exiting.")
            break

        first_pass = False

        # Choose randomly from the first N remaining scenarios, where N starts
        # with 1. If that one is already checked out, drop it from the candidate
        # pool, double N, wait a bit and try again with a broader search. We do
        # this instead of going through the list sequentially, to reduce the
        # chance of many workers contending to checkout the same next scenario.
        # With hundreds of workers, that can cause them to spend more time
        # trying to checkout a scenario than actually running it. On the other
        # hand, we start with small N in order to preferentially checkout the
        # earliest scenarios in the queue.

        to_run = list(scen_info.keys())
        N = 1
        rescan = False
        while to_run and not rescan:
            next_scen = random.randrange(min(N, len(to_run)))
            scenario_name = to_run[next_scen]
            if checkout(scenario_name):
                # run this scenario, then start again with a fresh scenario list
                scenario_args = (
                    scenario_option_file_args
                    + scen_info[scenario_name]
                    + scenario_cmd_line_args
                )
                rescan = True
                yield (scenario_name, scenario_args)
            else:
                logger.info(
                    f"Tried to checkout scenario {scenario_name} but it was already in use."
                )
                # drop this scenario from the candidate pool
                del to_run[next_scen]
                # decide whether to rescan or widen the search space for next time
                if N > 8 * len(to_run):
                    # we've had 3+ failed checkouts after expanding the search
                    # to the whole scenario list; try a fresher set of available
                    # scenarios (scanning the dir is ~200 times more expensive
                    # than trying a checkout, but may be needed once the pool is
                    # mostly or completely tapped out)
                    rescan = True
                else:
                    # ~double the search space and wait a bit to reduce
                    # contention. We approximately double rather than exactly, so
                    # that the pool of workers will be spread across various
                    # widths of search spaces, not stairstepped at 1, 2, 4, 8,
                    # etc. (may not matter)
                    N = int(N * (1.5 + random.random()) + 0.5)
                    time.sleep(random.random() * 0.5)

    # broke from the infinite loop because there were no more scenarios to run
    return


def parse_arg(arg, args=sys.argv[1:], **parse_kw):
    """Parse one argument from the argument list, using options as specified for argparse"""
    parser = _ArgumentParser(allow_abbrev=False)
    # Set output destination to 'option', so we can retrieve the value predictably.
    # This is done by updating parse_kw, so it can't be overridden by callers.
    # (They have no reason to set the destination anyway.)
    # note: we use the term "option" so that parsing errors will make a little more
    # sense, e.g., if users call with "--suffixes <blank>" (instead of just omitting it)
    parse_kw["dest"] = "option"
    parser.add_argument(arg, **parse_kw)
    return parser.parse_known_args(args)[0].option


def get_scenario_name(scenario_args):
    # use ad-hoc parsing to extract the scenario name from a scenario-definition string
    return parse_arg("--scenario-name", default=None, args=scenario_args)


def last_index(lst, val):
    try:
        return len(lst) - lst[::-1].index(val) - 1
    except ValueError:
        return -1


def read_scenario_list():
    # note: we read the list from the disk each time so that we get a fresher
    # version if the standard list is changed during a long solution effort.
    # This ignores comments in the scenario list (possibly starting mid-line),
    # just like switch solve does in options.txt.
    with open(scenario_list_file, "r") as f:
        scenario_list = [shlex.split(r, comments=True) for r in f.read().splitlines()]
    # drop any empty lines
    scenario_list = [s for s in scenario_list if s]
    return OrderedDict((get_scenario_name(s), s) for s in scenario_list)


def checkout(scenario_name, force=False):
    # write a flag that we are solving this scenario, before actually trying to lock it
    # this way, if the job gets interrupted in the middle of this function, the
    # worst that can happen is the scenario will be restarted then next time the job restarts
    # (if we locked the scenario and then got interrupted before setting the flag, then
    # the scenario would not be restarted when the job restarts, which is worse.)
    running_scenarios.append(scenario_name)
    write_running_scenarios_file()
    try:
        # create a lock directory for this scenario
        os.mkdir(os.path.join(scenario_queue_dir, scenario_name))
        locked = True
    except OSError as e:
        if e.errno != 17:  # File exists
            raise
        locked = False
    if locked or force:
        return True
    else:
        # remove the flag that we're running this scenario
        running_scenarios.remove(scenario_name)
        write_running_scenarios_file()
        return False


def mark_completed(scenario_name):
    # remove the scenario from the list of running scenarios (since it's been completed now)
    running_scenarios.remove(scenario_name)
    write_running_scenarios_file()
    # note: the scenario lock directory is left in place so the scenario won't get checked
    # out again


def write_running_scenarios_file():
    # write the list of scenarios currently being run by this job to disk
    # so they can be released back to the queue if the job is interrupted and restarted
    if running_scenarios:
        # note: we use open("r+") and truncate() instead of open("w")
        # to give a better chance of retaining the original file contents if this
        # job is interrupted in the middle of writing the file. This is not an issue unless
        # this job is running multiple scenarios at once
        # (If the file is only partial, then the queue will just think some jobs have been
        # done that actually haven't.)
        flags = "r+" if os.path.exists(running_scenarios_file) else "w"
        with open(running_scenarios_file, flags) as f:
            f.write("\n".join(running_scenarios) + "\n")
            f.truncate()
    else:
        # remove the running_scenarios_file entirely if it would be empty
        try:
            os.remove(running_scenarios_file)
        except OSError as e:
            if e.errno != 2:  # no such file
                raise


def unlock_running_scenarios():
    # called during startup to remove lockfiles for any scenarios that were still running
    # when this job was interrupted
    if os.path.exists(running_scenarios_file):
        with open(running_scenarios_file) as f:
            interrupted = f.read().splitlines()
        for scenario_name in interrupted:
            try:
                os.rmdir(os.path.join(scenario_queue_dir, scenario_name))
            except OSError as e:
                if e.errno != 2:  # no such file
                    raise


# run the main function if called as a script
if __name__ == "__main__":
    main()
