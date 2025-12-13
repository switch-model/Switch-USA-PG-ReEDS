"""
Minimize emissions subject to a budget cap from a previously solved model.
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
    # deactivate current objective and minimize discounted emissions instead
    for o in m.component_objects(Objective, active=True):
        o.deactivate()
    # use discounted emissions to give a fair balance between
    # earlier and later (i.e., to minimize discounted costs on a $/tCO2 basis)
    m.DiscountedEmissions = Objective(
        rule=lambda m: sum(
            m.AnnualEmissions[p] * m.bring_future_costs_to_base_year[p]
            for p in m.PERIODS
        ),
        sense=minimize,
    )


def define_dynamic_components(m):
    # enforce budget constraint (has to be done after SystemCost, which is
    # a dynamic component)
    with open(os.path.join(m.options.budget_dir, "total_cost.txt")) as f:
        budget = float(f.read().strip())
    m.BudgetLimit = Constraint(rule=lambda m: m.SystemCost <= budget)
