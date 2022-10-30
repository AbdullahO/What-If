import os
from typing import Tuple

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from numpy import ndarray

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


@pytest.fixture(scope="session")
def snn_model_matrix(snn_model: SNN) -> ndarray:
    X = snn_model.matrix
    assert X is not None
    return X


@pytest.fixture(scope="session")
def missing_set(snn_model_matrix: ndarray) -> ndarray:
    X = snn_model_matrix
    missing_set = np.argwhere(np.isnan(X))
    missing_row, missing_col = missing_set[0]
    assert np.isnan(X[missing_row, missing_col]), "first missing pair is not missing"
    return missing_set


@pytest.fixture(scope="session")
def example_missing_pair(missing_set: ndarray) -> ndarray:
    # Pick 20th missing pair because it has the first feasible output
    return missing_set[20]


@pytest.fixture(scope="session")
def example_obs_rows_and_cols(
    snn_model_matrix: ndarray, example_missing_pair: ndarray
) -> Tuple[frozenset, frozenset]:
    X = snn_model_matrix
    missing_row, missing_col = example_missing_pair
    example_obs_rows = frozenset(np.argwhere(~np.isnan(X[:, missing_col])).flatten())
    example_obs_cols = frozenset(np.argwhere(~np.isnan(X[missing_row, :])).flatten())
    return example_obs_rows, example_obs_cols


def test_predict(
    snn_model: SNN, snn_model_matrix: ndarray, example_missing_pair: ndarray
):
    prediction, feasible = snn_model._predict(snn_model_matrix, example_missing_pair)
    assert feasible, "prediction not feasible"
    assert prediction.round(6) == 48407.449874, "prediction has changed"


@pytest.mark.parametrize("k", [2, 4, 5])
def test_split(snn_model: SNN, test_anchor_rows: ndarray, k: int):
    anchor_rows_splits = list(snn_model._split(test_anchor_rows, k=k))
    quotient, remainder = divmod(len(test_anchor_rows), k)
    assert len(anchor_rows_splits) == k, "wrong number of splits"
    for idx, split in enumerate(anchor_rows_splits):
        expected_len = quotient + 1 if idx < remainder else quotient
        assert len(split) == expected_len


def test_model_repr(snn_model: SNN):
    assert str(snn_model) == (
        "SNN(linear_span_eps=0.1, max_rank=None, max_value=None,"
        " metric='sales', min_singular_value=1e-07, min_value=None,"
        " n_neighbors=1, random_splits=False, spectral_t=None, subspace_eps=0.1,"
        " verbose=False, weights='uniform')"
    )


def test_find_max_clique(
    snn_model_matrix: ndarray, example_obs_rows_and_cols: Tuple[frozenset, frozenset]
):
    obs_rows, obs_cols = example_obs_rows_and_cols
    _obs_rows = np.array(list(obs_rows), dtype=int)
    _obs_cols = np.array(list(obs_cols), dtype=int)

    # create bipartite incidence matrix
    X = snn_model_matrix
    B = X[_obs_rows]
    B = B[:, _obs_cols]
    assert np.any(np.isnan(B)), "B already fully connected"
    B[np.isnan(B)] = 0
    (n_rows, n_cols) = B.shape

    row_block_size = (n_rows, n_rows)
    col_block_size = (n_cols, n_cols)

    # No connections (all missing)
    A = np.block(
        [
            [np.ones(row_block_size), np.zeros_like(B)],
            [np.zeros_like(B.T), np.ones(col_block_size)],
        ]
    )
    G = nx.from_numpy_array(A)
    max_clique_rows_idx, max_clique_cols_idx = SNN._find_max_clique(G, n_rows)
    error_message = "Should return False for no clique"
    assert max_clique_rows_idx is False, error_message
    assert max_clique_cols_idx is False, error_message

    # real bipartite graph
    A = np.block([[np.ones(row_block_size), B], [B.T, np.ones(col_block_size)]])
    G = nx.from_numpy_array(A)
    max_clique_rows_idx, max_clique_cols_idx = SNN._find_max_clique(G, n_rows)
    error_message = "Should return ndarray"
    assert isinstance(max_clique_rows_idx, ndarray), error_message
    assert isinstance(max_clique_cols_idx, ndarray), error_message
    assert max_clique_rows_idx.shape == (35,)
    assert max_clique_cols_idx.shape == (10,)

    # Fully connected (none missing)
    A = np.block(
        [
            [np.ones(row_block_size), np.ones_like(B)],
            [np.ones_like(B.T), np.ones(col_block_size)],
        ]
    )
    G = nx.from_numpy_array(A)
    max_clique_rows_idx, max_clique_cols_idx = SNN._find_max_clique(G, n_rows)
    error_message = "Should return ndarray"
    assert isinstance(max_clique_rows_idx, ndarray), error_message
    assert isinstance(max_clique_cols_idx, ndarray), error_message
    assert max_clique_rows_idx.shape == (35,)
    assert max_clique_cols_idx.shape == (50,)
    # _obs_cols.shape == (50,)

#
# helper functions to test
# _get_anchors
# _spectral_rank
# _universal_rank
# _pcr
# _clip
# _train_error
# _subspace_inclusion
# _isfeasible

# _get_beta
# _synth_neighbor
# done _predict

# _get_tensor
# _check_input_matrix
# _prepare_input_data
# _check_weights
# done _split
# _find_anchors
# done _find_max_clique
