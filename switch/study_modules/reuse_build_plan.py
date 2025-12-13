"""
Solve model, reusing construction plan (BuildGen and BuildTx) from
previously run model with outputs saved in the directory specified
by --reuse-dir command line argument.
"""

import os
from pyomo.environ import *


def define_arguments(argparser):
    argparser.add_argument(
        "--reuse-dir",
        help="outputs-dir from previously run model; BuildGen and BuildTx values will be taken from here.",
    )


def define_components(m):
    m.build_gen_reused = Param(m.GEN_BLD_YRS, within=NonNegativeReals)
    m.Reuse_BuildGen = Constraint(
        m.GEN_BLD_YRS, rule=lambda m, g, p: m.BuildGen[g, p] == m.build_gen_reused[g, p]
    )
    m.build_tx_reused = Param(m.TRANS_BLD_YRS, within=NonNegativeReals)
    m.Reuse_BuildTx = Constraint(
        m.TRANS_BLD_YRS,
        rule=lambda m, tx, p: m.BuildTx[tx, p] == m.build_tx_reused[tx, p],
    )


def load_inputs(m, switch_data, inputs_dir):
    # treat saved versions of BuildGen and BuildTx like input files,
    # but renaming the columns
    switch_data.load_aug(
        filename=os.path.join(m.options.reuse_dir, "BuildGen.csv"),
        select=["GEN_BLD_YRS_1", "GEN_BLD_YRS_2", "BuildGen"],
        param=(m.build_gen_reused,),
    )
    switch_data.load_aug(
        filename=os.path.join(m.options.reuse_dir, "BuildTx.csv"),
        select=["TRANS_BLD_YRS_1", "TRANS_BLD_YRS_2", "BuildTx"],
        param=(m.build_tx_reused,),
    )
