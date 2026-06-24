from pyomo.environ import Constraint


def define_arguments(argparser):
    argparser.add_argument(
        "--add-solar-GW",
        "--add-solar-gw",
        dest="add_solar_gw",
        type=float,
        default=None,
        help="Minimum amount of utility-scale solar capacity to build in 2026 and later (GW).",
    )
    argparser.add_argument(
        "--total-solar-GW",
        "--total-solar-gw",
        dest="total_solar_gw",
        type=float,
        default=None,
        help="Minimum amount of utility-scale solar capacity to have in place during the study (GW).",
    )


# require at least the specified amount of utility-scale solar additions or capacity
def define_components(m):
    if m.options.add_solar_gw:
        m.Solar_Push_Additions = Constraint(
            rule=lambda m: sum(
                m.BuildGen[g, p]
                for g, p in m.BuildGen
                if p >= 2026
                and m.gen_energy_source[g] == "sun"
                and m.gen_is_distributed[g] == 0
            )
            >= m.options.add_solar_gw * 1000
        )
    if m.options.total_solar_gw:
        m.Solar_Push_Capacity = Constraint(
            rule=lambda m: sum(
                m.GenCapacity[g, p]
                for g, p in m.GenCapacity
                if m.gen_energy_source[g] == "sun" and m.gen_is_distributed[g] == 0
            )
            >= m.options.total_solar_gw * 1000
        )
