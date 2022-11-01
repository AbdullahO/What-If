import pandas as pd
import numpy as np
from synthetic_data_generation.syn_gyn_module import *


def get_sales_data():
    # Unit
    num_units = 100

    # Time
    num_timesteps = 50

    # Metrics
    metric1 = Metric("sales", metric_range=[0, 100000])
    metrics = [metric1]

    # Interventions: control(ad 0), (ad 1), (ad 2)
    num_interventions = 3

    # unit covariates: location and size
    loc = UnitCov("location", categories=["New York", "LA", "Boston"])
    size = UnitCov("size", categories=["small", "medium", "large"])
    unit_cov = [loc, size]

    # intervention covariates:
    # Note that the assignments here makes the association between ad 0 and intervention 00, ad 1 and intervention 1, etc ..
    treatment = IntCov(
        "ads",
        discrete=True,
        categories=["ad 0", "ad 1", "ad 2"],
        divisions=3,
        assignment=[0, 1, 2],
    )
    int_cov = [treatment]

    # initalize and generate
    data = SyntheticDataModule(
        num_units,
        num_timesteps,
        num_interventions,
        metrics,
        unit_cov,
        int_cov,
        freq="1D",
    )

    # generate initial data
    data.generate()

    # Now we will define differen subpopulations andd specific effects on them for each intervention

    # choose sub populations of interest where interventions will have different effects
    subpop1 = lambda: loc.unit_labels == "New York"
    subpop2 = lambda: loc.unit_labels == "Boston"
    subpop3 = lambda: loc.unit_labels == "LA"
    data.subpopulations_funcs = [subpop1, subpop2, subpop3]

    # We will assume that intervention 1 will incure an in crease of 30% in sales for some subpop, and 2 will incure a drop of 30% in sales for all subpops
    effects = [
        {"metric": metric1, "intervention": 1, "subpop": subpop1, "effect": 0.3},
        {"metric": metric1, "intervention": 1, "subpop": subpop2, "effect": 0.0},
        {"metric": metric1, "intervention": 1, "subpop": subpop3, "effect": 0.3},
        {"metric": metric1, "intervention": 2, "subpop": None, "effect": -0.4},
    ]

    data.add_effects(effects)

    return data


def sales_data_staggering_assignment(seed=0):

    np.random.seed(seed)
    data = get_sales_data()
    subpop1, subpop2, subpop3 = data.subpopulations_funcs
    num_timesteps = data.num_timesteps
    period_1 = {"intervention_assignment": "control", "until": 15}

    intervention_assignment = "cov_unit"
    selection_subpop = {
        subpop1: [0.3, 0.3, 0.4],
        subpop2: [1.0, 0.0, 0.0],
        subpop3: [1.0, 0.0, 0.0],
    }
    period_2 = {
        "intervention_assignment": intervention_assignment,
        "until": 30,
        "assignment_subpop": selection_subpop,
    }
    selection_subpop = {
        subpop1: [0.3, 0.3, 0.4],
        subpop1: [0.3, 0.3, 0.4],
        subpop3: [1.0, 0.0, 0.0],
    }
    period_3 = {
        "intervention_assignment": intervention_assignment,
        "until": 40,
        "assignment_subpop": selection_subpop,
    }
    selection_subpop = {
        subpop1: [0.3, 0.3, 0.4],
        subpop1: [0.3, 0.3, 0.4],
        subpop1: [0.3, 0.3, 0.4],
    }
    period_4 = {
        "intervention_assignment": intervention_assignment,
        "until": num_timesteps,
        "assignment_subpop": selection_subpop,
    }

    periods = [period_1, period_2, period_3, period_4]
    data.auto_subsample(periods)

    return data


def sales_data_random_assignment(seed=0):

    np.random.seed(seed)
    data = get_sales_data()
    num_timesteps = data.num_timesteps
    period_1 = {"intervention_assignment": "random", "until": num_timesteps}
    periods = [period_1]
    data.auto_subsample(periods)

    return data


def sales_data_si_assignment(seed=0):

    np.random.seed(seed)
    data = get_sales_data()
    subpop1, subpop2, subpop3 = data.subpopulations_funcs
    num_timesteps = data.num_timesteps
    period_1 = {"intervention_assignment": "control", "until": 20}
    intervention_assignment = "cov_unit"
    selection_subpop = {
        subpop1: [0.3, 0.3, 0.4],
        subpop2: [0.30, 0.30, 0.4],
        subpop3: [0.2, 0.4, 0.4],
    }
    period_2 = {
        "intervention_assignment": intervention_assignment,
        "until": num_timesteps,
        "assignment_subpop": selection_subpop,
    }
    periods = [period_1, period_2]
    data.auto_subsample(periods)

    return data
