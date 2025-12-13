"""
Minimize fossil power production subject to a budget cap from a previously solved model.
"""

# TODO: enforce the budget per-period and per-zone

import os
from pyomo.environ import *


def define_arguments(argparser):
    argparser.add_argument(
        "--budget-dir",
        help="outputs-dir from previously run model; total expenditure budget will be taken from here.",
    )


def define_components(m):
    # deactivate current objective and minimize fossil-based power production
    # instead
    for o in m.component_objects(Objective, active=True):
        o.deactivate()
    m.FuelBasedPower = Objective(
        rule=lambda m: sum(
            m.DispatchGen[g, tp] * m.tp_weight[tp]
            for g in m.FUEL_BASED_GENS
            for tp in m.TPS_FOR_GEN[g]
        ),
        sense=minimize,
    )


def define_dynamic_components(m):
    # enforce budget constraint (has to be done after SystemCost, which is
    # a dynamic component)
    with open(os.path.join(m.options.budget_dir, "total_cost.txt")) as f:
        budget = float(f.read().strip())
    m.BudgetLimit = Constraint(rule=lambda m: m.SystemCost <= budget)
