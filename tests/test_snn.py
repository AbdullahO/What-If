import os

import numpy as np
from numpy import ndarray
import pandas as pd
import pytest

from algorithms.snn import SNN

current_dir = os.getcwd()


@pytest.fixture(scope="session")
def test_anchor_rows() -> ndarray:
    # fmt: off
    anchor_rows = np.array([
        1,  2,  4,  9, 11, 16, 18, 19, 24, 25, 32, 33, 34, 37, 38, 49, 50,
       56, 57, 58, 59, 60, 63, 65, 67, 68, 80, 82, 84, 87, 90, 92, 96, 97,
       99
    ])
    # fmt: on
    return anchor_rows


@pytest.fixture(scope="session")
def snn_model() -> SNN:
    _snn_test_df = pd.read_csv(
        os.path.join(current_dir, "data/stores_sales_simple/stores_sales_simple.csv")
    )
    model = SNN(verbose=False)
    model.fit(
        df=_snn_test_df,
        unit_column="unit_id",
        time_column="time",
        metrics=["sales"],
        actions=["ads"],
    )
    return model


def test_predict(snn_model: SNN):
    X = snn_model.matrix
    missing_set = np.argwhere(np.isnan(X))
    assert np.isnan(X[0][1]), "first missing pair is not missing"
    # Pick 20th missing pair because it has the first feasible output
    missing_pair = missing_set[20]
    prediction, feasible = snn_model._predict(X, missing_pair)
    assert feasible, "prediction not feasible"
    assert prediction.round(6) == 48407.449874, "prediction has changed"


@pytest.mark.parametrize("k", [2, 4, 5])
def test_split(snn_model: SNN, test_anchor_rows: list, k: int):
    anchor_rows_splits = list(snn_model._split(test_anchor_rows, k=k))
    quotient, remainder = divmod(len(test_anchor_rows), k)
    assert len(anchor_rows_splits) == k, "wrong number of splits"
    for idx, split in enumerate(anchor_rows_splits):
        expected_len = quotient + 1 if idx < remainder else quotient
        assert len(split) == expected_len
