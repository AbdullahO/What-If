import warnings
from collections import namedtuple
from math import ceil

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
        default=["ALS"],
        help=f"choose algorithms from {ALG_REGISTRY.keys()}",
    )
    parser.add_argument(
        "--datasize",
        type=int,
        default=100,
        help=f"choose algorithms from {ALG_REGISTRY.keys()}",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="choose number of repetition for evaluation",
    )
    parser.add_argument(
        "--chunksize",
        nargs="+",
        type=int,
        default=[50],
        help="choose number of repetition for evaluation",
    )
    return parser


def evaluate_partial(data_gen, algorithm, repeat, datasize, chunk_size):
    # generate data
    train_time = np.zeros([repeat])
    query_time = np.zeros([repeat])
    r2 = np.zeros([repeat])
    mse = np.zeros([repeat])
    number_estimated_entries = np.zeros([repeat])
    number_estimated_feasible_entries = np.zeros([repeat])
    for i in range(repeat):
        data = data_gen(seed=i, num_units=200, num_timesteps=datasize)
        df = data.ss_df
        tensor = data.tensor
        mask = data.mask
        mask = mask.astype(bool)
        model = ALG_REGISTRY[algorithm](verbose=False)
        timestamps = pd.to_datetime(df.time.unique())
        no_batches = ceil(datasize / chunk_size)
        start = 0
        for batch in range(no_batches):
            start_time = timestamps[start]
            end = min(start + chunk_size - 1, datasize - 1)
            end_time = timestamps[end]
            df_batch = df.loc[(df.time >= start_time) & (df.time <= end_time)].copy()
            t = time.perf_counter()
            if batch == 0:
                model.fit(
                    df=df_batch,
                    unit_column="unit_id",
                    time_column="time",
                    metrics=["sales"],
                    actions=["ads"],
                )
            else:
                model.partial_fit(df_batch)
            train_time[i] = time.perf_counter() - t
            t = time.perf_counter()
            model.query(
                [0],
                ["2020-01-01", " 2020-01-02"],
                "sales",
                "ad 0",
                ["2020-01-01", " 2020-01-02"],
            )
            query_time[i] = time.perf_counter() - t
            start += chunk_size

        # adjust tensor
        indices = [model.actions_dict[action] for action in ["ad 0", "ad 1", "ad 2"]]
        _tensor_est = model.get_tensor_from_factors()
        tensor_est = _tensor_est[:, :, indices]
        # accuracy
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
            "chunksize",
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

    for k, data_gen in enumerate(datasets_generators[:]):
        for alg in args.algorithms:
            for chunk_size in args.chunksize:
                (
                    train_time[k, :],
                    query_time[k, :],
                    r2[k, :],
                    mse[k, :],
                    number_estimated_entries[k, :],
                    number_estimated_feasible_entries[k, :],
                ) = evaluate_partial(
                    data_gen, alg, args.repeat, args.datasize, chunk_size
                )
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
                    args.datasize,
                    chunk_size,
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
