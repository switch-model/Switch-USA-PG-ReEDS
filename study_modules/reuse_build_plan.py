"""
Reuse construction plan (BuildGen, BuildTx, etc.) from previously run model with
outputs saved in the directory specified by --reuse-dir command line argument.
"""

import os
from pyomo.environ import *

reuse_components = [
    "BuildGen",
    "BuildStorageEnergy",
    "SuspendGen",
    "BuildTx",
    "BuildLocalTD",
    "DeployEEShare",
    "DeployDRShare",
]


def define_arguments(argparser):
    argparser.add_argument(
        "--reuse-dir",
        help="""
            outputs-dir from previously run model; values for construction variables 
            will be taken from here.
        """,
    )


def define_dynamic_components(m):
    # Create Params like buildgen_reuse (BuildGen value from previous solution)
    # and Constraints like Reuse_BuildGen (force BuildGen[key] ==
    # buildgen_reuse[key]).
    # We define these late so the module load order doesn't matter.
    def make_reuse_rule(var, reuse_param):
        return lambda m, *key: getattr(m, var)[key] == getattr(m, reuse_param)[key]

    for var in reuse_components:
        comp = getattr(m, var, None)
        # note: bool(comp) is always False with Pyomo, so we have to be more specific
        if comp is not None:
            reuse_param = f"{var.lower()}_reused"
            reuse_constraint = f"Reuse_{var}"
            index_set = comp.index_set()
            setattr(m, reuse_param, Param(index_set, within=Reals))
            setattr(
                m,
                reuse_constraint,
                Constraint(index_set, rule=make_reuse_rule(var, reuse_param)),
            )


def load_inputs(m, switch_data, inputs_dir):
    # treat saved versions of construction vars like input files
    for var in reuse_components:
        reuse_file = os.path.join(m.options.reuse_dir, f"{var}.csv")
        comp = getattr(m, var, None)
        if comp is not None:
            index_set = comp.index_set()
            # copied from switch_model.reporting.save_component_values()
            cols = [f"{index_set.name}_{i+1}" for i in range(index_set.dimen)] + [var]
            switch_data.load_aug(
                filename=reuse_file,
                select=cols,
                param=(getattr(m, f"{var.lower()}_reused"),),
            )
        elif os.path.exists(reuse_file):
            raise RuntimeError(
                f"unable to reuse plan from {reuse_file} because "
                f"{var} is not defined in this model"
            )
