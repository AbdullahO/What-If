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
def expected_anchor_rows() -> ndarray:
    # fmt: off
    anchor_rows = np.array([
        1,  2,  4,  9, 11, 16, 18, 19, 24, 25, 32, 33, 34, 37, 38, 49, 50,
       56, 57, 58, 59, 60, 63, 65, 67, 68, 80, 82, 84, 87, 90, 92, 96, 97,
       99
    ])
    # fmt: on
    return anchor_rows


@pytest.fixture(scope="session")
def expected_anchor_cols() -> ndarray:
    anchor_cols = np.array([0, 3, 6, 9, 12, 15, 18, 21, 24, 27])
    return anchor_cols


@pytest.fixture(scope="session")
def expected_train_error() -> float:
    expected_error = 0.00024697
    return expected_error


@pytest.fixture(scope="session")
def expected_s_rank() -> float:
    s_rank = 899188.86289752
    return s_rank


@pytest.fixture(scope="session")
def expected_pred_synth_neighbor() -> float:
    pred = 48407.44987403
    return pred


@pytest.fixture(scope="session")
def expected_u_rank():
    u_rank = np.array(
        [
            [-0.39175393],
            [-0.3800858],
            [-0.32084485],
            [-0.20323565],
            [-0.14651091],
            [-0.16050234],
            [-0.24560417],
            [-0.34898293],
            [-0.41527478],
            [-0.39500305],
        ]
    )
    return u_rank


@pytest.fixture(scope="session")
def expected_v_rank() -> ndarray:
    # fmt: off
    v_rank = np.array([[
        -0.17010884, -0.1645733 , -0.18563122, -0.17592975, -0.19599812,
        -0.18437652, -0.16344502, -0.16365546, -0.16050917, -0.18532319,
        -0.1629684 , -0.17552058, -0.15424275, -0.1629496 , -0.15859282,
        -0.19301049, -0.17922021, -0.16809427, -0.14787918, -0.16423993,
        -0.16046308, -0.15716652, -0.1557234 , -0.16724974, -0.17699057,
        -0.17173043, -0.16232822, -0.1704932 , -0.18230142, -0.16803178,
        -0.16982261, -0.1414294 , -0.16470539, -0.17186123, -0.16534298
    ]])
    # fmt: on
    assert v_rank.shape == (1, 35)
    return v_rank


@pytest.fixture(scope="session")
def expected_beta() -> ndarray:
    # fmt: off
    beta = np.array([
        0.03070154, 0.02970248, 0.03350305, 0.03175211, 0.03537408,
        0.0332766 , 0.02949884, 0.02953682, 0.02896898, 0.03344745,
        0.02941282, 0.03167826, 0.027838  , 0.02940943, 0.02862311,
        0.03483487, 0.03234598, 0.03033795, 0.02668949, 0.02964231,
        0.02896066, 0.02836569, 0.02810523, 0.03018553, 0.03194357,
        0.03099421, 0.02929728, 0.03077091, 0.03290208, 0.03032667,
        0.03064988, 0.02552543, 0.02972632, 0.03101782, 0.02984139
    ])
    # fmt: on
    assert beta.shape == (35,)
    return beta


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
def snn_model_matrix_full(snn_model: SNN) -> ndarray:
    X_full = snn_model.matrix_full
    assert X_full is not None
    return X_full


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
def example_X1_y1(
    snn_model: SNN,
    snn_model_matrix: ndarray,
    expected_anchor_rows: ndarray,
    expected_anchor_cols: ndarray,
    example_missing_pair: ndarray,
) -> Tuple[ndarray, ndarray]:
    # TODO: rename X1 and y1 here and in snn.py?
    missing_row, _missing_col = example_missing_pair
    y1 = snn_model_matrix[missing_row, expected_anchor_cols]
    X1 = snn_model_matrix[expected_anchor_rows, :]
    X1 = X1[:, expected_anchor_cols]
    return X1, y1


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
    """Test the _predict function"""
    prediction, feasible = snn_model._predict(snn_model_matrix, example_missing_pair)
    assert feasible, "prediction not feasible"
    assert prediction.round(6) == 48407.449874, "prediction has changed"


@pytest.mark.parametrize("k", [2, 4, 5])
def test_split(snn_model: SNN, expected_anchor_rows: ndarray, k: int):
    anchor_rows_splits = list(snn_model._split(expected_anchor_rows, k=k))
    quotient, remainder = divmod(len(expected_anchor_rows), k)
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


def test_get_anchors(
    snn_model: SNN,
    snn_model_matrix: ndarray,
    snn_model_matrix_full: ndarray,
    example_obs_rows_and_cols: ndarray,
    expected_anchor_rows: ndarray,
    expected_anchor_cols: ndarray,
):
    """Test the _get_anchors function"""
    snn_model._get_anchors.cache.clear()
    obs_rows, obs_cols = example_obs_rows_and_cols
    anchor_rows, anchor_cols = snn_model._get_anchors(
        snn_model_matrix, obs_rows, obs_cols
    )

    error_message = "Anchor rows not as expected"
    assert np.allclose(anchor_rows, expected_anchor_rows), error_message
    error_message = "Anchor columns not as expected"
    assert np.allclose(anchor_cols, expected_anchor_cols), error_message

    _obs_rows = np.array(list(obs_rows), dtype=int)
    _obs_cols = np.array(list(obs_cols), dtype=int)
    B = snn_model_matrix_full[_obs_rows]
    B = B[:, _obs_cols]
    assert not np.any(np.isnan(B)), "snn_model_matrix_full contains NaN"

    # Test matrix_full, which should short circuit and return the input
    snn_model._get_anchors.cache.clear()
    anchor_rows, anchor_cols = snn_model._get_anchors(
        snn_model_matrix_full, obs_rows, obs_cols
    )

    error_message = "Anchor rows not as expected"
    assert np.allclose(anchor_rows, _obs_rows), error_message
    error_message = "Anchor columns not as expected"
    assert np.allclose(anchor_cols, _obs_cols), error_message


def test_find_anchors(
    snn_model: SNN,
    snn_model_matrix: ndarray,
    example_missing_pair: ndarray,
    expected_anchor_rows: ndarray,
    expected_anchor_cols: ndarray,
):
    """Test the _find_anchors function"""
    snn_model._get_anchors.cache.clear()
    anchor_rows, anchor_cols = snn_model._find_anchors(
        snn_model_matrix, example_missing_pair
    )

    error_message = "Anchor rows not as expected"
    assert np.allclose(anchor_rows, expected_anchor_rows), error_message
    error_message = "Anchor columns not as expected"
    assert np.allclose(anchor_cols, expected_anchor_cols), error_message


@pytest.mark.parametrize("spectral_t", [0.9, 0.7, 0.5])
def test_spectral_rank(
    snn_model: SNN, example_X1_y1: Tuple[ndarray, ndarray], spectral_t: float
):
    """Test the _spectral_rank function"""
    X1, _y1 = example_X1_y1
    U, S, V = np.linalg.svd(X1.T, full_matrices=False)
    snn_model.spectral_t = spectral_t
    rank = snn_model._spectral_rank(S)
    assert rank == 1, f"spectral rank at spectral_t = {spectral_t} not as expected"


def test_universal_rank(
    snn_model: SNN, snn_model_matrix: ndarray, example_X1_y1: Tuple[ndarray, ndarray]
):
    """Test the _universal_rank function"""

    X1, _y1 = example_X1_y1
    U, S, V = np.linalg.svd(X1.T, full_matrices=False)
    m, n = snn_model_matrix.shape
    ratio = m / n
    snn_model.spectral_t = None
    rank = snn_model._universal_rank(S, ratio=ratio)
    assert rank == 3, "universal rank not as expected"


def test_pcr(
    snn_model: SNN,
    example_X1_y1: Tuple[ndarray, ndarray],
    expected_beta: ndarray,
    expected_u_rank: ndarray,
    expected_v_rank: ndarray,
    expected_s_rank: float,
):
    """Test the _pcr function"""
    X1, y1 = example_X1_y1
    beta, u_rank, s_rank, v_rank = snn_model._pcr(X1.T, y1)
    assert np.allclose(beta, expected_beta), "beta not as expected"
    assert s_rank.shape == (1,), "s_rank shape not as expected"
    assert s_rank[0].round(8) == expected_s_rank
    assert np.allclose(u_rank, expected_u_rank), "u_rank not as expected"
    assert np.allclose(v_rank, expected_v_rank), "v_rank not as expected"


def test_train_error(
    snn_model: SNN,
    example_X1_y1: Tuple[ndarray, ndarray],
    expected_beta: ndarray,
    expected_train_error: ndarray,
):
    """Test the _train_error function"""
    X1, y1 = example_X1_y1
    train_error = snn_model._train_error(X1.T, y1, expected_beta)
    assert train_error.round(8) == expected_train_error, "train_error not as expected"


def test_get_beta(
    snn_model: SNN,
    snn_model_matrix: ndarray,
    example_missing_pair: ndarray,
    expected_anchor_rows: ndarray,
    expected_anchor_cols: ndarray,
    expected_beta: ndarray,
    expected_v_rank: ndarray,
    expected_train_error: float,
):
    """Test the _get_beta function"""
    snn_model._get_beta.cache.clear()
    missing_row, _missing_col = example_missing_pair
    _anchor_rows = frozenset(expected_anchor_rows)
    _anchor_cols = frozenset(expected_anchor_cols)
    beta, v_rank, train_error = snn_model._get_beta(
        snn_model_matrix, missing_row, _anchor_rows, _anchor_cols
    )
    assert np.allclose(beta, expected_beta), "beta not as expected"
    assert np.allclose(v_rank, expected_v_rank), "v_rank not as expected"
    assert train_error.round(8) == expected_train_error, "train_error not as expected"


def test_clip(snn_model: SNN):
    """Test the _clip function"""
    snn_model.min_value = 1
    snn_model.max_value = 5
    assert snn_model._clip(11) == 5, "should clip to max_value"
    assert snn_model._clip(0) == 1, "should clip to min_value"
    assert snn_model._clip(-10) == 1, "should clip to min_value"
    # Put back for other tests. TODO: use a context manager?
    snn_model.min_value = None
    snn_model.max_value = None


def test_synth_neighbor(
    snn_model: SNN,
    snn_model_matrix: ndarray,
    example_missing_pair: ndarray,
    expected_anchor_rows: ndarray,
    expected_anchor_cols: ndarray,
    expected_pred_synth_neighbor: float,
):
    """Test the _synth_neighbor function"""
    pred, feasible, weight = snn_model._synth_neighbor(
        snn_model_matrix,
        example_missing_pair,
        expected_anchor_rows,
        expected_anchor_cols,
    )
    assert pred.round(8) == expected_pred_synth_neighbor, "pred not as expected"
    assert feasible, "feasible should be True"
    assert weight == 1.0, "weight not as expected"


def test_get_tensor():
    """Test the _get_tensor function"""


def test_check_input_matrix():
    """Test the _check_input_matrix function"""


def test_prepare_input_data():
    """Test the _prepare_input_data function"""


def test_check_weights():
    """Test the _check_weights function"""
