# do minimal reporting to reduce disk usage for multi-year weekly models


def define_components(m):
    allowed_post_solve = {
        # These modules report info that is needed even for daily/weekly models, but some
        # also have their own flags to reduce the output more finely
        "study_modules.generators_core_dispatch",
        "switch_model.generators.core.dispatch",
        "switch_model.reporting",
        "study_modules.unserved_load",
        "study_modules.planning_reserves_extreme_days",
        "study_modules.report_zonal_dispatch",
    }
    for module in m.get_modules():
        if hasattr(module, "post_solve") and module.__name__ not in allowed_post_solve:
            del module.post_solve
