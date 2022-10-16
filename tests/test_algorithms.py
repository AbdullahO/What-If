import os
import sys

import numpy as np
import pandas as pd
import pytest

current_dir = os.getcwd()  # whatIf folder
curr_algos_path = os.path.join(current_dir, "current_algorithms/si")
sys.path.append(curr_algos_path)
from current_algorithms.si import SI as current_SI

# import algorithms.si as si


@pytest.fixture(scope="session")
def basque_pre_df() -> pd.DataFrame:
    pre_df = pd.read_pickle(os.path.join(current_dir, "data/basque_pre_df.pkl"))
    return pre_df


@pytest.fixture(scope="session")
def basque_post_df() -> pd.DataFrame:
    post_df = pd.read_pickle(os.path.join(current_dir, "data/basque_post_df.pkl"))
    return post_df


@pytest.fixture(scope="session")
def basque_cross_validation_df() -> pd.DataFrame:
    cross_validation_df = pd.read_pickle(
        os.path.join(current_dir, "data/basque_cross_validation_df.pkl")
    )
    return cross_validation_df


@pytest.fixture(scope="session")
def basque_synthetic_results_df() -> pd.DataFrame:
    synthetic_results_df = pd.read_pickle(
        os.path.join(current_dir, "data/basque_synthetic_results_df.pkl")
    )
    return synthetic_results_df


def test_current_synthetic_interventions(
    basque_pre_df,
    basque_post_df,
    basque_cross_validation_df,
    basque_synthetic_results_df,
):
    basque = current_SI.SI(t=0.99, center=False)
    basque.fit(basque_pre_df, basque_post_df)
    cross_validation_score = basque.cross_validation_score

    r2_rct_col = "R2_rct scores"
    r2_col = "R2 scores"

    expected_r2_rct = basque_cross_validation_df[r2_rct_col]
    expected_r2 = basque_cross_validation_df[r2_col]
    assert np.allclose(
        expected_r2_rct, cross_validation_score[r2_rct_col], equal_nan=True
    ), "R2_rct not as expected"
    assert np.allclose(
        expected_r2, cross_validation_score[r2_col], equal_nan=True
    ), "R2 not as expected"

    assert basque.synthetic_results.equals(
        basque_synthetic_results_df
    ), "Synthetic results not as expected"
