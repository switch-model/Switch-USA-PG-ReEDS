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


def define_dynamic_components(m):
    # we define these late so the module load order doesn't matter
    m.build_gen_reused = Param(m.GEN_BLD_YRS, within=NonNegativeReals)
    m.Reuse_BuildGen = Constraint(
        m.GEN_BLD_YRS, rule=lambda m, g, p: m.BuildGen[g, p] == m.build_gen_reused[g, p]
    )
    m.suspend_gen_reused = Param(m.GEN_BLD_SUSPEND_YRS, within=NonNegativeReals)

    def rule(m, g, v, p):
        result = m.SuspendGen[g, v, p] == m.suspend_gen_reused[g, v, p]
        if (g, v, p) == ("p123_petroleum_liquids_1", 1529, 2030):
            breakpoint()
        return result

    m.Reuse_SuspendGen = Constraint(
        m.GEN_BLD_SUSPEND_YRS,
        rule=rule,
    )
    # m.Reuse_SuspendGen = Constraint(
    #     m.GEN_BLD_SUSPEND_YRS,
    #     rule=lambda m, g, v, p: m.SuspendGen[g, v, p] == m.suspend_gen_reused[g, v, p],
    # )
    m.build_tx_reused = Param(m.TRANS_BLD_YRS, within=NonNegativeReals)
    m.Reuse_BuildTx = Constraint(
        m.TRANS_BLD_YRS,
        rule=lambda m, tx, p: m.BuildTx[tx, p] == m.build_tx_reused[tx, p],
    )
    if hasattr(m, "DeployEEShare"):
        m.deploy_ee_reused = Param(m.LOAD_ZONES, m.PERIODS, within=PercentFraction)
        m.Reuse_DeployEEShare = Constraint(
            m.LOAD_ZONES,
            m.PERIODS,
            rule=lambda m, z, p: m.DeployEEShare[z, p] == m.deploy_ee_reused[z, p],
        )
    if hasattr(m, "DeployDRShare"):
        m.deploy_dr_reused = Param(m.LOAD_ZONES, m.PERIODS, within=PercentFraction)
        m.Reuse_DeployDRShare = Constraint(
            m.LOAD_ZONES,
            m.PERIODS,
            rule=lambda m, z, p: m.DeployDRShare[z, p] == m.deploy_dr_reused[z, p],
        )


def load_inputs(m, switch_data, inputs_dir):
    # treat saved versions of construction vars like input files,
    # but renaming the columns
    switch_data.load_aug(
        filename=os.path.join(m.options.reuse_dir, "BuildGen.csv"),
        select=["GEN_BLD_YRS_1", "GEN_BLD_YRS_2", "BuildGen"],
        param=(m.build_gen_reused,),
    )
    switch_data.load_aug(
        filename=os.path.join(m.options.reuse_dir, "SuspendGen.csv"),
        select=[
            "GEN_BLD_SUSPEND_YRS_1",
            "GEN_BLD_SUSPEND_YRS_2",
            "GEN_BLD_SUSPEND_YRS_3",
            "SuspendGen",
        ],
        param=(m.suspend_gen_reused,),
    )
    switch_data.load_aug(
        filename=os.path.join(m.options.reuse_dir, "BuildTx.csv"),
        select=[
            "TRANS_BLD_YRS_1",
            "TRANS_BLD_YRS_2",
            "BuildTx",
        ],
        param=(m.build_tx_reused,),
    )
    if hasattr(m, "DeployEEShare"):
        switch_data.load_aug(
            filename=os.path.join(m.options.reuse_dir, "DeployEEShare.csv"),
            select=[
                "SetProduct_OrderedSet_1",
                "SetProduct_OrderedSet_2",
                "DeployEEShare",
            ],
            param=(m.deploy_ee_reused,),
        )
    if hasattr(m, "DeployDRShare"):
        switch_data.load_aug(
            filename=os.path.join(m.options.reuse_dir, "DeployDRShare.csv"),
            select=[
                "SetProduct_OrderedSet_1",
                "SetProduct_OrderedSet_2",
                "DeployDRShare",
            ],
            param=(m.deploy_dr_reused,),
        )
