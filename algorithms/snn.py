"""
Synthetic Nearest Neighbors algorithm:
Refactoring the code in https://github.com/deshen24/syntheticNN
"""
import sys
from typing import Any, Iterator, Optional, Tuple
import networkx as nx
import numpy as np
from cachetools import cached
from cachetools.keys import hashkey
from networkx.algorithms.clique import find_cliques  # type: ignore
from numpy import float64, int64, ndarray
from algorithms.fill_tensor_base import FillTensorBase
import sparse


class SNN(FillTensorBase):
    """
    Impute missing entries in a matrix via SNN algorithm
    """

    def __init__(
        self,
        n_neighbors: int = 1,
        weights: str = "uniform",
        random_splits: bool = False,
        max_rank: Optional[int] = None,
        spectral_t: Optional[float] = None,
        linear_span_eps: float = 0.1,
        subspace_eps: float = 0.1,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        verbose: bool = True,
        min_singular_value: float = 1e-7,
        full_training_time_steps:int = 10,
        threshold_multiplier:int = 10, 
        L: Optional[int] = None, 
        k_factors: Optional[int] = 5
    ) -> None:
        """
        Parameters
        ----------
        n_neighbors : int
        Number of synthetic neighbors to construct

        weights : str
        Weight function used in prediction. Possible values:
        (a) 'uniform': each synthetic neighbor is weighted equally
        (b) 'distance': weigh points inversely with distance (as per train error)

        random_splits : bool
        Randomize donors prior to splitting

        max_rank : int
        Perform truncated SVD on training data with this value as its rank

        spectral_t : float
        Perform truncated SVD on training data with (100*thresh)% of spectral energy retained.
        If omitted, then the default value is chosen via Donoho & Gavish '14 paper.

        linear_span_eps : float
        If the (normalized) train error is greater than (100*linear_span_eps)%,
        then the missing pair fails the linear span test.

        subspace_eps : float
        If the test vector (used for predictions) does not lie within (100*subspace_eps)% of
        the span covered by the training vectors (used to build the model),
        then the missing pair fails the subspace inclusion test.

        min_value : float
        Minumum possible imputed value

        max_value : float
        Maximum possible imputed value

        verbose : bool
        """
        super().__init__(verbose=verbose, min_singular_value=min_singular_value, 
                         full_training_time_steps = full_training_time_steps,
                         threshold_multiplier = threshold_multiplier, L = L,
                         k_factors = k_factors)
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.random_splits = random_splits
        self.max_rank = max_rank
        self.spectral_t = spectral_t
        self.linear_span_eps = linear_span_eps
        self.subspace_eps = subspace_eps
        self.min_value = min_value
        self.max_value = max_value

    def diagnostics(self):
        """returns method-specifc diagnostics"""
        raise NotImplementedError()

    def summary(self):
        """returns method-specifc summary"""
        raise NotImplementedError()

    def _initialize(self, X: ndarray, missing_set: ndarray) -> Tuple[ndarray, ndarray]:
        # check and prepare data
        X = self._prepare_input_data(X, missing_set, 2)
        # check weights
        self.weights = self._check_weights(self.weights)
        # initialize
        X_imputed = X.copy()
        self.feasible = np.empty(X.shape)
        self.feasible.fill(np.nan)
        return X, X_imputed

    def _fit_transform(self, X: ndarray, test_set: Optional[ndarray] = None) -> ndarray:
        """
        complete missing entries in matrix
        """
        # 
        
        filled_matrix = self._snn_fit_transform(X, test_set)
        # reshape matrix into tensor
        
        return filled_matrix

    def _snn_fit_transform(
        self, X: ndarray, test_set: Optional[ndarray] = None
    ) -> ndarray:
        """
        complete missing entries in matrix
        """
        # tensor to matrix

        missing_set = test_set
        if missing_set is None:
            missing_set = np.argwhere(np.isnan(X))
        num_missing = len(missing_set)

        X, X_imputed = self._initialize(X, missing_set)

        # complete missing entries
        for (i, missing_pair) in enumerate(missing_set):
            if self.verbose:
                print("[SNN] iteration {} of {}".format(i + 1, num_missing))

            # predict missing entry
            (pred, feasible) = self._predict(X, missing_pair=missing_pair)
            # store in imputed matrices
            (missing_row, missing_col) = missing_pair
            X_imputed[missing_row, missing_col] = pred
            self.feasible[missing_row, missing_col] = feasible

        if self.verbose:
            print("[SNN] complete")
        return X_imputed

    def _check_weights(self, weights: str) -> str:
        """
        check to make sure weights are valid
        """
        if weights not in ("uniform", "distance"):
            raise ValueError(
                "weights not recognized: should be 'uniform' or 'distance'"
            )
        return weights

    def _split(self, arr: ndarray, k: int) -> Iterator[Any]:
        """
        split array arr into k subgroups of roughly equal size
        """
        (m, n) = divmod(len(arr), k)
        return (arr[i * m + min(i, n) : (i + 1) * m + min(i + 1, n)] for i in range(k))

    def _find_anchors(
        self, X: ndarray, missing_pair: ndarray
    ) -> Tuple[ndarray, ndarray]:
        """
        find model learning submatrix by reducing to max biclique problem
        """
        (missing_row, missing_col) = missing_pair

        obs_rows = frozenset(np.argwhere(~np.isnan(X[:, missing_col])).flatten())
        obs_cols = frozenset(np.argwhere(~np.isnan(X[missing_row, :])).flatten())
        return self._get_anchors(X, obs_rows, obs_cols)

    @staticmethod
    def _find_max_clique(G: nx.Graph, n_rows: int):
        cliques = list(find_cliques(G))
        d_min = 0
        max_clique_rows_idx = False
        max_clique_cols_idx = False
        for clique in cliques:
            clique = np.sort(clique)
            clique_rows_idx = clique[clique < n_rows]
            clique_cols_idx = clique[clique >= n_rows] - n_rows
            d = min(len(clique_rows_idx), len(clique_cols_idx))
            if d > d_min:
                d_min = d
                max_clique_rows_idx = clique_rows_idx
                max_clique_cols_idx = clique_cols_idx
        return max_clique_rows_idx, max_clique_cols_idx

    @cached(
        cache=dict(),  # type: ignore
        key=lambda self, X, obs_rows, obs_cols: hashkey(obs_rows, obs_cols),
    )
    def _get_anchors(
        self, X: ndarray, obs_rows: frozenset, obs_cols: frozenset
    ) -> Tuple[ndarray, ndarray]:
        _obs_rows = np.array(list(obs_rows), dtype=int)
        _obs_cols = np.array(list(obs_cols), dtype=int)
        # create bipartite incidence matrix
        B = X[_obs_rows]
        B = B[:, _obs_cols]
        if not np.any(np.isnan(B)):  # check if fully connected already
            return (_obs_rows, _obs_cols)
        B[np.isnan(B)] = 0

        # bipartite graph
        (n_rows, n_cols) = B.shape
        A = np.block([[np.ones((n_rows, n_rows)), B], [B.T, np.ones((n_cols, n_cols))]])
        G = nx.from_numpy_array(A)

        # find max clique that yields the most square (nxn) matrix
        max_clique_rows_idx, max_clique_cols_idx = SNN._find_max_clique(G, n_rows)

        # determine model learning rows & cols
        anchor_rows = _obs_rows[max_clique_rows_idx]
        anchor_cols = _obs_cols[max_clique_cols_idx]
        return (anchor_rows, anchor_cols)

    def _spectral_rank(self, s):
        """
        retain all singular values that compose at least (100*self.spectral_t)% spectral energy
        """
        if self.spectral_t == 1.0:
            rank = len(s)
        else:
            total_energy = (s**2).cumsum() / (s**2).sum()
            rank = list((total_energy > self.spectral_t)).index(True) + 1
        return rank

    def _universal_rank(self, s: ndarray, ratio: float) -> int:
        """
        retain all singular values above optimal threshold as per Donoho & Gavish '14:
        https://arxiv.org/pdf/1305.5870.pdf
        """
        omega = 0.56 * ratio**3 - 0.95 * ratio**2 + 1.43 + 1.82 * ratio
        t = omega * np.median(s)
        rank = max(len(s[s > t]), 1)
        return rank

    def _pcr(self, X: ndarray, y: ndarray) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        """
        principal component regression (PCR)
        """
        (u, s, v) = np.linalg.svd(X, full_matrices=False)
        # filter out small singular values
        s[s < self.min_singular_value] = 0

        if self.max_rank is not None:
            rank = self.max_rank
        elif self.spectral_t is not None:
            rank = self._spectral_rank(s)
        else:
            (m, n) = X.shape
            rank = self._universal_rank(s, ratio=m / n)

        s_rank = s[:rank]
        u_rank = u[:, :rank]
        v_rank = v[:rank, :]

        beta = ((v_rank.T / s_rank) @ u_rank.T) @ y
        return (beta, u_rank, s_rank, v_rank)

    def _clip(self, x: float64) -> float64:
        """
        clip values to fall within range [min_value, max_value]
        """
        if self.min_value is not None:
            min_value = np.float64(self.min_value)
            x = max(min_value, x)
        if self.max_value is not None:
            max_value = np.float64(self.max_value)
            x = min(max_value, x)
        return x

    def _train_error(self, X: ndarray, y: ndarray, beta: ndarray) -> float64:
        """
        compute (normalized) training error
        """
        y_pred = X @ beta
        delta = np.linalg.norm(y_pred - y)
        ratio = delta / np.linalg.norm(y)
        return ratio**2

    def _subspace_inclusion(self, V1: ndarray, X2: ndarray) -> float64:
        """
        compute subspace inclusion statistic
        """
        delta = (np.eye(V1.shape[1]) - (V1.T @ V1)) @ X2
        ratio = np.linalg.norm(delta) / np.linalg.norm(X2)
        return ratio**2

    def _isfeasible(
        self, train_error: float64, subspace_inclusion_stat: float64
    ) -> bool:
        """
        check feasibility of prediction
        True iff linear span + subspace inclusion tests both pass
        """
        # linear span test
        ls_feasible = True if train_error <= self.linear_span_eps else False
        # subspace test
        s_feasible = True if subspace_inclusion_stat <= self.subspace_eps else False
        return ls_feasible and s_feasible

    @cached(
        cache=dict(),  # type: ignore
        key=lambda self, X, missing_row, anchor_rows, anchor_cols: hashkey(
            missing_row, anchor_rows, anchor_cols
        ),
    )
    def _get_beta(
        self,
        X: ndarray,
        missing_row: int64,
        anchor_rows: frozenset,
        anchor_cols: frozenset,
    ) -> Tuple[ndarray, ndarray, float64]:
        _anchor_rows = np.array(list(anchor_rows), dtype=int)
        _anchor_cols = np.array(list(anchor_cols), dtype=int)
        y1 = X[missing_row, _anchor_cols]
        X1 = X[_anchor_rows, :]
        X1 = X1[:, _anchor_cols]
        (beta, _, s_rank, v_rank) = self._pcr(X1.T, y1)
        train_error = self._train_error(X1.T, y1, beta)
        return beta, v_rank, train_error

    def _synth_neighbor(
        self,
        X: ndarray,
        missing_pair: ndarray,
        anchor_rows: ndarray,
        anchor_cols: ndarray,
    ) -> Tuple[float64, bool, float]:
        """
        construct the k-th synthetic neighbor
        """
        # initialize
        (missing_row, missing_col) = missing_pair
        X2 = X[anchor_rows, missing_col]

        # learn k-th synthetic neighbor
        _anchor_rows = frozenset(anchor_rows)
        _anchor_cols = frozenset(anchor_cols)

        beta, v_rank, train_error = self._get_beta(
            X, missing_row, _anchor_rows, _anchor_cols
        )
        # prediction
        pred = self._clip(X2 @ beta)

        # diagnostics
        subspace_inclusion_stat = self._subspace_inclusion(v_rank, X2)
        feasible = self._isfeasible(train_error, subspace_inclusion_stat)

        # assign weight of k-th synthetic neighbor
        if self.weights == "uniform":
            weight = 1.0
        elif self.weights == "distance":
            d = float(train_error + subspace_inclusion_stat)
            weight = (1.0 / d) if d > 0 else sys.float_info.max
        return (pred, feasible, weight)

    def _update_nan_mask(self, new_tensor):

        assert (
            self.tensor_nans is not None
        ), "self.tensor_nans is None, have you called fit()?"
        old_shape = self.tensor_nans.shape
        self.tensor_nans.shape = (
            old_shape[0],
            new_tensor.shape[1] + old_shape[1],
            old_shape[2],
        )
        # remove unobserved (i,t) pairs
        N, T, I = new_tensor.shape
        X = new_tensor.reshape([N, I * T])
        mask = np.isnan(X).sum(0)
        missing = np.where((mask >= N - 3))[0]
        missing_timesteps = missing // I
        missing_actions = missing % I
        tensor_nans = self.tensor_nans.todense()
        for t, i in zip(missing_timesteps, missing_actions):
            tensor_nans[:, old_shape[1] + t, i] = 1
        self.tensor_nans = sparse.COO.from_numpy(tensor_nans)

    def _predict(self, X: ndarray, missing_pair: ndarray) -> Tuple[float64, bool]:
        """
        combine predictions from all synthetic neighbors
        """
        # find anchor rows and cols
        (anchor_rows, anchor_cols) = self._find_anchors(X, missing_pair=missing_pair)
        if not anchor_rows.size:
            (pred, feasible) = (np.float64(np.nan), False)
        else:
            if self.random_splits:
                anchor_rows = np.random.permutation(anchor_rows)
            anchor_rows_splits = list(self._split(anchor_rows, k=self.n_neighbors))
            pred_arr = np.zeros(self.n_neighbors)
            feasible_arr = np.zeros(self.n_neighbors)
            weights = np.zeros(self.n_neighbors)
            # iterate through all row splits
            for (k, anchor_rows_k) in enumerate(anchor_rows_splits):
                _pred, _feasible, _weight = self._synth_neighbor(
                    X,
                    missing_pair=missing_pair,
                    anchor_rows=anchor_rows_k,
                    anchor_cols=anchor_cols,
                )
                pred_arr[k] = _pred
                feasible_arr[k] = _feasible
                weights[k] = _weight
            weights /= np.sum(weights)
            pred = np.float64(np.average(pred_arr, weights=weights))
            feasible = all(feasible_arr)
        return (pred, feasible)
