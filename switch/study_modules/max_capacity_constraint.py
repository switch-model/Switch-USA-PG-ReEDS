"""
Implement requirement of maximum installed capacity for certain type of energy
source in some areas. -- etc, offshore wind


max_cap_requirement.csv shows the requirement capacity.
max_cap_generators.csv has the list of qualified generators.

"""

import os
from pyomo.environ import (
    Set,
    Param,
    Expression,
    Constraint,
    Suffix,
    NonNegativeReals,
    Reals,
    Any,
    Var,
)

from switch_model.utilities import unique_list


def define_components(m):

    # (program, period) combinations with maximum capacity rules in effect
    m.MAX_CAP_RULES = Set(dimen=2, within=Any * m.PERIODS)
    # maximum capacity specified for each (program, period) combination
    m.max_cap_mw = Param(m.MAX_CAP_RULES, within=Reals)
    # set of all maximum-capacity programs
    m.MAX_CAP_PROGRAMS = Set(
        initialize=lambda m: unique_list(pr for pr, pe in m.MAX_CAP_RULES)
    )

    # set of all valid program/generator combinations (i.e., gens participating
    # in each program)
    m.MAX_CAP_PROGRAM_GENS = Set(within=m.MAX_CAP_PROGRAMS * m.GENERATION_PROJECTS)
    m.GENS_IN_MAX_CAP_PROGRAM = Set(
        m.MAX_CAP_PROGRAMS,
        within=m.GENERATION_PROJECTS,
        initialize=lambda m, pr: unique_list(
            _g for (_pr, _g) in m.MAX_CAP_PROGRAM_GENS if _pr == pr
        ),
    )

    # enforce constraint on total installed capacity in each program in each period
    def rule(m, pr, pe):
        if not m.GENS_IN_MAX_CAP_PROGRAM[pr]:
            # program may have no participating generators,
            # in which case the constraint is always met
            return Constraint.Skip

        build_capacity = sum(
            m.GenCapacity[g, pe] for g in m.GENS_IN_MAX_CAP_PROGRAM[pr]
        )
        max_capacity_requirement = m.max_cap_mw[pr, pe]

        # define and return the constraint
        return build_capacity <= max_capacity_requirement

    m.Enforce_Max_Capacity = Constraint(m.MAX_CAP_RULES, rule=rule)


def load_inputs(model, switch_data, inputs_dir):
    """
    Expected input files:
    max_cap_generators.csv
        MAX_CAP_PROGRAM,PERIOD,MAX_CAP_GEN

    max_cap_requirements.csv
        MAX_CAP_PROGRAM,PERIOD,max_cap_mw

    """
    switch_data.load_aug(
        filename=os.path.join(inputs_dir, "max_cap_requirements.csv"),
        optional=True,  # also enables empty files
        index=model.MAX_CAP_RULES,
        param=(model.max_cap_mw,),
    )
    switch_data.load_aug(
        filename=os.path.join(inputs_dir, "max_cap_generators.csv"),
        optional=True,  # also enables empty files
        set=model.MAX_CAP_PROGRAM_GENS,
    )
