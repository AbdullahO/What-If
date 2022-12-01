import warnings
from collections import namedtuple

warnings.simplefilter(action="ignore", category=FutureWarning)
import pandas as pd
import numpy as np
import argparse
from algorithms.snn import SNN
from algorithms.snn_biclustering import SNNBiclustering
from algorithms.fill_tensor_ALS import ALS
from sklearn.metrics import r2_score
import time
from synthetic_data_generation.generate_eval import (
    sales_data_staggering_assignment,
    sales_data_si_assignment,
    sales_data_random_assignment,
    get_sales_data,
)

Metric = namedtuple(
    "Metric",
    "train_time  query_time r2 mse number_estimated_entries number_estimated_feasible_entries",
)
ALG_REGISTRY = {"SNN": SNN, "SNNBiclustering": SNNBiclustering, "ALS": ALS}


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", action="store_true", help="export data")
    parser.add_argument(
        "--algorithms",
        nargs="+",
        type=str,
        default=["SNN"],
        help=f"choose algorithms from {ALG_REGISTRY.keys()}",
    )
    parser.add_argument(
        "--datasize",
        nargs="+",
        type=int,
        default=[100],
        help=f"choose algorithms from {ALG_REGISTRY.keys()}",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="choose number of repetition for evaluation",
    )
    return parser


def evaluate(data_gen, data_assignment, algorithm, repeat, datasize):
    # generate data
    train_time = np.zeros([repeat])
    query_time = np.zeros([repeat])
    r2 = np.zeros([repeat])
    mse = np.zeros([repeat])
    number_estimated_entries = np.zeros([repeat])
    number_estimated_feasible_entries = np.zeros([repeat])
    for i in range(repeat):
        data = data_gen(seed=i, N=datasize, T=datasize)

        tensor, full_df = data.generate([0, datasize - 1])
        periods = data_assignment(data, seed=i, T=data.max_timesteps)
        _, df = data.auto_subsample(periods, tensor, full_df)

        mask = data.mask
        mask = mask.astype(bool)
        model = ALG_REGISTRY[algorithm](verbose=False)
        t = time.perf_counter()
        model.fit(
            df=df,
            unit_column="unit_id",
            time_column="time",
            metrics=["sales"],
            actions=["ads"],
        )
        train_time[i] = time.perf_counter() - t
        t = time.perf_counter()
        model.query(
            [0],
            ["2020-01-10", " 2020-01-10"],
            "sales",
            "ad 0",
            ["2020-01-10", " 2020-01-10"],
        )
        query_time[i] = time.perf_counter() - t

        # adjust tensor
        indices = [model.actions_dict[action] for action in ["ad 0", "ad 1", "ad 2"]]
        _tensor_est = model.get_tensor_from_factors()
        tensor_est = _tensor_est[:, :, indices]
        # accuracy ()
        notnan = ~np.isnan(tensor_est)
        r2[i] = r2_score(
            tensor[notnan, 0][~mask[notnan, 0]].flatten(),
            tensor_est[notnan][~mask[notnan, 0]].flatten(),
        )
        mse[i] = np.nanmean(
            np.square(
                tensor[..., 0][~mask[..., 0]].flatten()
                - tensor_est[:][~mask[..., 0]].flatten()
            )
        )
        number_estimated_entries[i] = (~np.isnan(tensor_est[:][~mask[..., 0]])).sum()
        number_estimated_feasible_entries[i] = np.nansum(model.feasible)

        # model._get_anchors.cache.clear()
        # model._get_beta.cache.clear()
    return Metric(
        train_time,
        query_time,
        r2,
        mse,
        number_estimated_entries,
        number_estimated_feasible_entries,
    )


def main():
    parser = create_parser()
    args = parser.parse_args()

    # data examples
    datasets_generators = [
        sales_data_si_assignment,
        sales_data_staggering_assignment,
        sales_data_random_assignment,
    ]
    data_names = ["SI sparsity", "staggered", "random"]
    num_datasets = len(datasets_generators)

    # init results dataframe
    res_df = pd.DataFrame(
        columns=[
            "datasize",
            "algorithm",
            "data name",
            "train_time (sec)",
            "query_time (ms)",
            "R2",
            "RMSE",
            "number of estimated entries",
        ]
    )

    # init metrics
    train_time = np.zeros([num_datasets, args.repeat])
    query_time = np.zeros([num_datasets, args.repeat])
    r2 = np.zeros([num_datasets, args.repeat])
    mse = np.zeros([num_datasets, args.repeat])
    number_estimated_entries = np.zeros([num_datasets, args.repeat])
    number_estimated_feasible_entries = np.zeros([num_datasets, args.repeat])
    data_gen = get_sales_data
    for k, data_assignment in enumerate(datasets_generators):
        for alg in args.algorithms:
            for datasize in args.datasize:
                (
                    train_time[k, :],
                    query_time[k, :],
                    r2[k, :],
                    mse[k, :],
                    number_estimated_entries[k, :],
                    number_estimated_feasible_entries[k, :],
                ) = evaluate(data_gen, data_assignment, alg, args.repeat, datasize)
                print(f"Evaluate {alg} for {data_names[k]}")
                print(f"Train time: \t {train_time[k,:].mean()}")
                print(f"query time: \t {query_time[k,:].mean()}")
                print(f"R2: \t {r2[k,:].mean()}")
                print(f"RMSE: \t {np.sqrt(mse)[k,:].mean()}")
                print(
                    f"number of retrieved entries: \t {number_estimated_entries[k,:].mean()}"
                )
                print(
                    f"number of feasible retrieved entries: \t {number_estimated_feasible_entries[k,:].mean()}"
                )
                print("=" * 100)
                res_df.loc[res_df.shape[0]] = [
                    datasize,
                    alg,
                    data_names[k],
                    train_time[k, :].mean(),
                    query_time[k, :].mean(),
                    r2[k, :].mean(),
                    np.sqrt(mse)[k, :].mean(),
                    number_estimated_entries[k, :].mean(),
                ]
                res_df.to_csv("test_metrics.csv")

    print("summary:")
    print(res_df)

    if args.export:
        res_df.to_csv("test_metrics.csv")


if __name__ == "__main__":
    main()
