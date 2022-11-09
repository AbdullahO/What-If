import os
from typing import Tuple

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from numpy import ndarray

from algorithms.snn import SNN
from algorithms.als import AlternatingLeastSquares as ALS

current_dir = os.getcwd()


@pytest.fixture(scope="session")
def snn_expected_query_output() -> pd.DataFrame:
    query_output = pd.read_pickle(
        os.path.join(current_dir, "data/stores_sales_simple/snn_query_output.pkl")
    )
    return query_output


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
    expected_error = 0.0
    return expected_error


@pytest.fixture(scope="session")
def expected_s_rank() -> float:
    s_rank = 899188.86289752
    return s_rank


@pytest.fixture(scope="session")
def expected_pred_synth_neighbor() -> float:
    pred = 43785.38590169
    return pred


@pytest.fixture(scope="session")
def expected_u_rank():
    u_rank = np.array(
        [
            [-0.39175393, -0.22962289, 0.593236],
            [-0.3800858, -0.16013307, 0.16595897],
            [-0.32084485, 0.05929045, -0.29865917],
            [-0.20323565, 0.42009182, -0.26491154],
            [-0.14651091, 0.55613296, 0.23675001],
            [-0.16050234, 0.50950738, 0.28005658],
            [-0.24560417, 0.28462043, -0.20686294],
            [-0.34898293, -0.04259613, -0.10781215],
            [-0.41527478, -0.21732567, -0.49287033],
            [-0.39500305, -0.20664686, 0.17127201],
        ]
    )

    return u_rank


@pytest.fixture(scope="session")
def expected_v_rank() -> ndarray:
    # fmt: off
    v_rank = np.array([
        [
            -0.17010884, -0.1645733, -0.18563122, -0.17592975, -0.19599812, -0.18437652,
            -0.16344502, -0.16365546, -0.16050917, -0.18532319, -0.1629684, -0.17552058,
            -0.15424275, -0.1629496, -0.15859282, -0.19301049, -0.17922021, -0.16809427,
            -0.14787918, -0.16423993, -0.16046308, -0.15716652, -0.1557234, -0.16724974,
            -0.17699057, -0.17173043, -0.16232822, -0.1704932, -0.18230142, -0.16803178,
            -0.16982261, -0.1414294, -0.16470539, -0.17186123, -0.16534298
        ],
        [
            0.11814578, 0.25223188, -0.02862758, -0.20286332, -0.10803572, -0.17638829,
            0.07146581, -0.01720852, 0.08832767, -0.16389963, 0.1251174, 0.02858141,
            0.15708663, 0.19742302, 0.09383668, 0.07779065, 0.3311298, 0.0057554,
            0.14609831, -0.21360775, -0.22632754, -0.15400062, -0.14977987, 0.02192565,
            -0.21208115, -0.07005689, 0.23984759, -0.225828, -0.11327426, -0.0526727,
            0.1245144, 0.44182979, -0.05107998, -0.20189541, 0.00108859
        ],
        [
            -0.05866279, -0.02713989, -0.24056099, -0.01838087, -0.36963921, -0.16074381,
            0.06417162, 0.09706598, 0.10302096, -0.18058313, 0.04970982, -0.10641876,
            0.17257583, 0.02050765, 0.13062201, -0.39890746, -0.2874579, 0.01856149,
            0.27617444, 0.1680775, 0.23209283, 0.25393504, 0.27469072, 0.02511942,
            -0.03114371, -0.00714837, 0.01288018, 0.07566417, -0.15416838, 0.04336879,
            -0.05680253, 0.25599879, 0.09452953, 0.04459374, 0.06331815
        ],
    ])
    # fmt: on
    assert v_rank.shape == (3, 35)
    return v_rank


@pytest.fixture(scope="session")
def expected_beta() -> ndarray:
    # fmt: off
    beta = np.array([ 
        0.04552686,  0.04314364,  0.07720297,  0.02840542,  0.10038835,  0.05723781,
        0.01996807,  0.01094581,  0.01278863,  0.06150737,  0.02435395,  0.05238426,
        0.00103744,  0.03218009,  0.00750291,  0.111473,    0.09673675,  0.02708425,
        -0.01970745, -0.00867333, -0.02166148, -0.02390122, -0.02787379,  0.02625326,
        0.03065886,  0.02998709,  0.0348972,   0.0092022,   0.05774545,  0.02052157,
        0.04534207, -0.00727385,  0.01047767,  0.01601383,  0.01812434,
    ])
    # fmt: on
    assert beta.shape == (35,)
    return beta


def get_store_sales_simple_df() -> pd.DataFrame:
    snn_test_df = pd.read_csv(
        os.path.join(current_dir, "data/stores_sales_simple/stores_sales_simple.csv")
    )
    return snn_test_df


@pytest.fixture(scope="session")
def snn_model() -> SNN:
    snn_test_df = get_store_sales_simple_df()
    model = SNN(verbose=False)
    model.fit(
        df=snn_test_df,
        unit_column="unit_id",
        time_column="time",
        metrics=["sales"],
        actions=["ads"],
    )
    return model


@pytest.fixture(scope="session")
def snn_model_matrix(snn_model: SNN) -> ndarray:
    unit_column = "unit_id"
    time_column = "time"
    actions = ["ads"]
    metrics = ["sales"]
    df = get_store_sales_simple_df()
    # convert time to datetime column
    df[time_column] = pd.to_datetime(df[time_column])
    # get tensor and dimensions
    tensor, N, I, T = snn_model._get_tensor(
        df, unit_column, time_column, actions, metrics
    )
    matrix = tensor.reshape([N, I * T])
    return matrix


@pytest.fixture(scope="session")
def snn_model_matrix_full(snn_model: SNN) -> ndarray:
    N = snn_model.N
    T = snn_model.T
    I = snn_model.I
    tensor = snn_model.get_tensor_from_factors()
    matrix_full = tensor.reshape([N, I * T])
    return matrix_full


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
    assert prediction.round(6) == 43785.385902, "prediction has changed"


@pytest.mark.parametrize("k", [2, 4, 5])
def test_split(snn_model: SNN, expected_anchor_rows: ndarray, k: int):
    anchor_rows_splits = list(snn_model._split(expected_anchor_rows, k=k))
    quotient, remainder = divmod(len(expected_anchor_rows), k)
    assert len(anchor_rows_splits) == k, "wrong number of splits"
    for idx, split in enumerate(anchor_rows_splits):
        expected_len = quotient + 1 if idx < remainder else quotient
        assert len(split) == expected_len


def test_model_str(snn_model: SNN):
    assert str(snn_model) == (
        "SNN(I=3, N=100, T=50, linear_span_eps=0.1, max_rank=None, max_value=None,"
        " metric='sales', min_singular_value=1e-07, min_value=None,"
        " n_neighbors=1, random_splits=False, spectral_t=None, subspace_eps=0.1,"
        " verbose=False, weights='uniform')"
    )


def test_model_repr(snn_model: SNN):
    assert repr(snn_model) == str(snn_model)


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
    snn_model._get_anchors.cache.clear()  # type: ignore
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
    assert not np.any(np.isnan(B)), "snn_model_matrix_full: B contains NaN"

    # Test matrix_full, which should short circuit and return the input
    snn_model._get_anchors.cache.clear()  # type: ignore
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
    snn_model._get_anchors.cache.clear()  # type: ignore
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
    S[S < snn_model.min_singular_value] = 0
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
    assert s_rank.shape == (3,), "s_rank shape not as expected"
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
    snn_model._get_beta.cache.clear()  # type: ignore
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
    assert snn_model._clip(np.float64(11.0)) == 5, "should clip to max_value"
    assert snn_model._clip(np.float64(0.1)) == 1, "should clip to min_value"
    assert snn_model._clip(np.float64(-10.5)) == 1, "should clip to min_value"
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


def get_new_snn_model_pre_fit() -> Tuple:
    model = SNN(verbose=False)
    unit_column = "unit_id"
    time_column = "time"
    actions = ["ads"]
    metrics = ["sales"]
    model.metric = metrics[0]
    df = get_store_sales_simple_df()
    # convert time to datetime column
    df[time_column] = pd.to_datetime(df[time_column])
    # get tensor and dimensions
    tensor, N, I, T = model._get_tensor(df, unit_column, time_column, actions, metrics)
    return df, model, tensor, N, I, T


def test_get_tensor(snn_model_matrix: ndarray):
    """Test the _get_tensor function"""
    # Don't use snn_model fixture here
    _df, _model, tensor, N, I, T = get_new_snn_model_pre_fit()
    matrix = tensor.reshape([N, I * T])
    error_message = "_get_tensor matrix output not as expected"
    assert np.allclose(matrix, snn_model_matrix, equal_nan=True), error_message


def test_fit_transform(snn_model_matrix_full: ndarray, snn_model_matrix: ndarray):
    """Test the _fit_transform function"""
    # Don't use snn_model fixture here
    df, model, tensor, N, I, T = get_new_snn_model_pre_fit()

    # tensor to matrix
    matrix = tensor.reshape([N, I * T])
    error_message = "_get_tensor matrix output not as expected"
    assert np.allclose(matrix, snn_model_matrix, equal_nan=True), error_message
    snn_imputed_matrix = model._fit_transform(matrix)
    error_message = "_fit_transform output shape not as expected"
    assert snn_imputed_matrix.shape == (100, 150), error_message

    # Check that we get the same ALS output
    tensor = snn_imputed_matrix.reshape([N, T, I])
    nans_mask = np.isnan(tensor)
    als_model = ALS()
    # Modifies tensor by filling nans with zeros
    als_model.fit(tensor)
    # Reconstructs tensor from CP form, all values now filled
    tensor = als_model.predict()
    tensor[nans_mask] = np.nan
    matrix_full = tensor.reshape([N, I * T])

    error_message = "_fit_transform output not as expected"
    assert np.allclose(
        matrix_full, snn_model_matrix_full, equal_nan=True
    ), error_message


def test_check_input_matrix(snn_model: SNN, snn_model_matrix: ndarray):
    """Test the _check_input_matrix function"""
    missing_mask = np.argwhere(np.isnan(snn_model_matrix))

    # Should not raise any exceptions
    snn_model._check_input_matrix(snn_model_matrix, missing_mask)

    # Should raise an error due to missing_mask length
    m, n = snn_model_matrix.shape
    with pytest.raises(ValueError):
        snn_model._check_input_matrix(snn_model_matrix, np.ones(m * n))

    # Should raise an error due to incorrect shape of snn_model_matrix
    new_shape = (100, 50, 3)
    with pytest.raises(ValueError):
        snn_model._check_input_matrix(snn_model_matrix.reshape(new_shape), missing_mask)


def test_prepare_input_data(snn_model: SNN, snn_model_matrix: ndarray):
    """Test the _prepare_input_data function"""
    missing_mask = np.argwhere(np.isnan(snn_model_matrix))
    output = snn_model._prepare_input_data(snn_model_matrix, missing_mask)
    error_message = "_prepare_input_data output not as expected"
    assert np.allclose(output, snn_model_matrix, equal_nan=True), error_message


def test_initialize(snn_model: SNN, snn_model_matrix: ndarray):
    """Test the _initialize function"""
    # TODO: should rename to missing_mask in _initialize?
    missing_set = np.argwhere(np.isnan(snn_model_matrix))
    X, X_imputed = snn_model._initialize(snn_model_matrix, missing_set)
    error_message = "_initialize output not as expected"
    assert np.allclose(X, snn_model_matrix, equal_nan=True), error_message


@pytest.mark.parametrize("weights", ["uniform", "distance", "something else", ""])
def test_check_weights(snn_model: SNN, weights):
    """Test the _check_weights function"""
    if weights in ("uniform", "distance"):
        output = snn_model._check_weights(weights)
        assert output == weights
    else:
        with pytest.raises(ValueError):
            snn_model._check_weights(weights)


def test_fit(snn_model: SNN, snn_expected_query_output: pd.DataFrame):
    model_query_output = snn_model.query(
        [0],
        ["2020-01-10", " 2020-02-19"],
        "sales",
        "ad 0",
        ["2020-01-10", " 2020-02-19"],
    )
    assert snn_expected_query_output.round(5).equals(
        model_query_output.round(5)
    ), "Query output difference"
    assert snn_model.actions_dict == {
        "ad 2": 0,
        "ad 0": 1,
        "ad 1": 2,
    }, "Actions dict difference"
