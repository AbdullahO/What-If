import argparse
from ast import arg
import pandas as pd
import numpy as np
from syn_gyn_module import SyntheticDataModule, Metric, UnitCov, IntCov


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", action="store_true", help="export data")
    parser.add_argument("--seed", type=int, default=0, help="choose seed")
    args = parser.parse_args()

    np.random.seed(args.seed)
    # Unit
    num_units = 300

    # Time
    num_timesteps = 300

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
    period_1 = {"intervention_assignment": "control", "until": 100}
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
    if args.export:
        data.export("store_sales_simple_2", dir="data/")

    return data


if __name__ == "__main__":
    data = main()
