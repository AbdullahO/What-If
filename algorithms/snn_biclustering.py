"""
Synthetic Nearest Neighbors algorithm Variant with biclustering to speed up computation
A varaint of the algorithm in  https://github.com/deshen24/syntheticNN
"""
import warnings
import pandas as pd
import numpy as np
from sklearn.utils import check_array  # type: ignore
from sklearn.cluster import SpectralBiclustering
from tensorly.decomposition import parafac
from algorithms.snn import SNN
from cachetools import cached
from cachetools.keys import hashkey


## TODO: clear cache function


class SNNBiclustering(SNN):
    """
    Impute missing entries in a matrix via SNN algorithm + biclustering
    """

    def __init__(
        self,
        random_splits=False,
        max_rank=None,
        spectral_t=None,
        linear_span_eps=0.1,
        subspace_eps=0.1,
        min_value=None,
        max_value=None,
        verbose=True,
        min_singular_value=1e-7,
        min_row_sparsity=0.3,
        min_col_sparsity=0.3,
        min_cluster_sparsity=0.3,
        min_cluster_size=5,
        no_clusterings=3,
        min_num_clusters=5,
        num_estimates=3,
        seed=None,
    ):
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

        min_singular_value : float
        minimum singular value to include when learning the linear model (for numerical stability)

        min_row_sparsity : float
        minimum sparsity level (should b in (0.1.0]) for rows to be included in the cluster

        min_col_sparsity : float
        minimum sparsity level (should b in (0.1.0]) for cols to be included in the cluster

        min_cluster_sparsity : float
        minimum sparsity level (should b in (0.1.0]) for cluster to be included
        """
        super().__init__()
        self.random_splits = random_splits
        self.max_rank = max_rank
        self.spectral_t = spectral_t
        self.linear_span_eps = linear_span_eps
        self.subspace_eps = subspace_eps
        self.min_value = min_value
        self.max_value = max_value
        self.verbose = verbose
        self.min_singular_value = min_singular_value
        self.min_col_sparsity = min_col_sparsity
        self.min_row_sparsity = min_row_sparsity
        self.min_cluster_sparsity = min_cluster_sparsity
        self.clusters = None
        self.clusters_row_matrix = None
        self.clusters_col_matrix = None
        self.min_cluster_size = min_cluster_size
        self.no_clusterings = no_clusterings
        self.min_num_clusters = min_num_clusters
        self.num_estimates = num_estimates
        self.clusters_hashes = None
        self.seed = seed
        if self.seed is None:
            self.seed = np.random.randint(100)

    def _filter_cluster(self, cluster_mask, rows, cols):
        # if very very sparse, drop
        if cluster_mask.mean() < 0.2:
            return
        # remove columns and then rows that are mostly nans
        col_avages = cluster_mask.mean(0)
        retained_cols = col_avages > self.min_col_sparsity
        retained_cols_index = cols[retained_cols]
        cluster_mask = cluster_mask[:, retained_cols]

        # if no columns, return
        if cluster_mask.shape[1] == 0:
            return

        row_avages = cluster_mask.mean(1)
        retained_rows = row_avages > self.min_row_sparsity
        retained_rows_index = rows[retained_rows]
        cluster_mask = cluster_mask[retained_rows, :]

        # if now rows, return
        if cluster_mask.shape[0] == 0:
            return

        # check sparsity, if more than min_cluster_sparsity drop
        if cluster_mask.mean() < self.min_cluster_sparsity:
            return
        return retained_rows_index, retained_cols_index

    def _filter_and_construct_cluster(self, cluster, X, mask, rows, cols):
        filtered_cluster_rows_and_cols = self._filter_cluster(cluster, rows, cols)
        ## TODO: change into class
        if filtered_cluster_rows_and_cols is None:
            return
        rows, cols = filtered_cluster_rows_and_cols
        cluster = {
            "rows": rows,
            "cols": cols,
            "data": X[rows, :][:, cols],
            "sparsity": 1 - mask[rows, :][:, cols].mean(),
        }
        return cluster

    def _get_clusters(self):
        self.clusters = {}
        index = 0
        self.clusters_hashes = set()

        for idx in range(self.no_clusterings):
            model = SpectralBiclustering(
                n_clusters=(
                    self.min_num_clusters + 2 * idx,
                    self.min_num_clusters + 2 * idx,
                ),
                n_best=6 + idx,
                n_components=7 + idx,
                method="log",
                random_state=idx + self.seed,
                mini_batch=True,
            )
            model.fit(self.mask)

            ## process clusters
            row_clusters = np.unique(model.row_labels_)
            col_clusters = np.unique(model.column_labels_)
            # loop over clusters
            for row_clus in row_clusters:
                for col_clus in col_clusters:
                    rows = np.where(model.row_labels_ == row_clus)[0]
                    cols = np.where(model.column_labels_ == col_clus)[0]
                    set_ = set(rows)
                    set_.update(cols + self.mask.shape[0])
                    hash_ = hash(frozenset(set_))
                    if hash_ in self.clusters_hashes:
                        continue
                    else:
                        self.clusters_hashes.update({hash_})
                    cluster = np.array(self.mask[rows, :][:, cols])
                    cluster = self._filter_and_construct_cluster(
                        cluster, self.matrix, self.mask, rows, cols
                    )

                    if cluster is None:
                        continue
                    self.clusters[index] = cluster
                    index += 1

        # if self.verbose:
        print(f"Generated {index} clusters")

    def _get_clusters_matrices(self):
        clusters_row_matrix = np.zeros([len(self.clusters), self.matrix.shape[0]])
        for i, cl in enumerate(self.clusters.values()):
            clusters_row_matrix[i, cl["rows"]] = 1
        clusters_col_matrix = np.zeros([len(self.clusters), self.matrix.shape[1]])
        for i, cl in enumerate(self.clusters.values()):
            clusters_col_matrix[i, cl["cols"]] = 1
        return clusters_row_matrix, clusters_col_matrix

    def _fit_transform(self, X, test_set=None):
        """
        complete missing entries in matrix
        """
        # tensor to matrix
        self.matrix = X
        self.mask = (~np.isnan(X)).astype(int)

        # get clusters:
        self.min_num_clusters = int(np.sqrt(min(self.mask.shape)))
        self._get_clusters()

        # set cluster matrices for finding best clusters
        (
            self.clusters_row_matrix,
            self.clusters_col_matrix,
        ) = self._get_clusters_matrices()

        return super()._snn_fit_transform(X, test_set)

    def _predict(self, X, missing_pair):
        i, j = missing_pair
        obs_rows = np.argwhere(~np.isnan(X[:, j])).flatten()
        obs_cols = np.argwhere(~np.isnan(X[i, :])).flatten()

        if not obs_rows.size or not obs_cols.size:
            return np.nan, False

        estimates = np.full(self.num_estimates, np.nan)
        feasibles = np.full(self.num_estimates, np.nan)

        # construct row vector
        obs_rows_vector = np.zeros(X.shape[0])
        obs_rows_vector[obs_rows] = 1
        obs_cols_vector = np.zeros(X.shape[1])
        obs_cols_vector[obs_cols] = 1

        # multiply by cluster_row_matrix and clusters_col_matrix
        clusters_row_matching = self.clusters_row_matrix @ obs_rows_vector
        clusters_col_matching = self.clusters_col_matrix @ obs_cols_vector

        # select based on max
        selected_cluster = np.argsort(clusters_row_matching * clusters_col_matching)[
            -self.num_estimates :
        ]

        # if there are not big enough cluster
        if max(clusters_row_matching * clusters_col_matching) < self.min_cluster_size:
            return np.nan, False

        counter = 0
        for clus in selected_cluster:
            # get minimal anchor rows and cols
            rows_cluster = self.clusters_row_matrix[clus, :]
            cols_cluster = self.clusters_col_matrix[clus, :]
            anchor_rows = np.where(obs_rows_vector * rows_cluster)[0]
            anchor_cols = np.where(obs_cols_vector * cols_cluster)[0]

            cluster = np.array(self.mask[anchor_rows, :][:, anchor_cols])
            if not cluster.any():
                continue

            if (
                cluster.sum(0).min() < 2
                or cluster.mean(1).sum() < 2
                or cluster.mean() < 0.3
            ):
                continue

            prediction, feasible = self._synth_neighbor(
                X, missing_pair, anchor_rows, anchor_cols, clus
            )
            estimates[counter] = prediction
            feasibles[counter] = feasible

            counter += 1
        if not counter:
            return np.nan, False
        else:
            return np.nanmean(estimates), bool(np.nanmax(feasibles))

    def _synth_neighbor(self, X, missing_pair, anchor_rows, anchor_cols, cluster_idx):
        """
        estimate the missing values based on cluster <cluster?
        """
        # check if factors is already computed
        (missing_row, missing_col) = missing_pair
        cluster = self.clusters[cluster_idx]
        if "u_rank" not in cluster:
            cluster = self._get_factors(X, cluster)

        _anchor_rows = frozenset(anchor_rows)
        _anchor_cols = frozenset(anchor_cols)

        beta, u_rank, train_error = self._get_beta_from_factors(
            X, missing_row, _anchor_rows, _anchor_cols, cluster_idx
        )
        X2 = X[anchor_rows, missing_col]
        # prediction
        pred = X2 @ beta

        # diagnostics
        subspace_inclusion_stat = self._subspace_inclusion(u_rank, X2)
        feasible = self._isfeasible(train_error, subspace_inclusion_stat)
        return pred, feasible

    def _get_factors(self, X, cluster):
        cluster_rows = cluster["rows"]
        cluster_cols = cluster["cols"]
        cluster["rows_dict"] = dict(zip(cluster_rows, np.arange(len(cluster_rows))))
        cluster["cols_dict"] = dict(zip(cluster_cols, np.arange(len(cluster_cols))))
        X1 = X[cluster_rows, :]
        X1 = X1[:, cluster_cols]
        (u_rank, s_rank, v_rank) = self._compute_factors(X1)
        cluster["u_rank"] = u_rank
        cluster["v_rank"] = v_rank
        cluster["s_rank"] = s_rank
        return cluster

    def _als(self, X):
        X_copy = np.array(X)
        X_copy[np.isnan(X_copy)] = 0
        p = 1 - np.isnan(X_copy).sum() / X_copy.size
        (_, s, _) = np.linalg.svd(X_copy / p, full_matrices=False)
        if self.max_rank is not None:
            rank = self.max_rank
        elif self.spectral_t is not None:
            rank = self._spectral_rank(s)
        else:
            (m, n) = X.shape
            rank = self._universal_rank(s, ratio=m / n)
        rank = min(np.sum(s > self.min_singular_value), rank)
        weights, factors = parafac(
            X_copy,
            rank=rank,
            mask=~np.isnan(X),
            init="random",
            normalize_factors=True,
        )
        ## this is a quick solution to make sure factors are orthogonal -- only needed for diagnosis
        # (u_n, s1, v1) = factors[0], weights, factors[1]
        (u_n, s1, v1) = np.linalg.svd(
            weights * factors[0] @ factors[1].T, full_matrices=False
        )
        return u_n, s1, v1.T

    def _compute_factors(self, X):
        """
        principal component regression (PCR)
        """
        u_rank, s_rank, v_rank = self._als(X)
        return u_rank, s_rank, v_rank

    @cached(
        cache=dict(),  # type: ignore
        key=lambda self, X, missing_row, anchor_rows, anchor_cols, cluster_idx: hashkey(
            missing_row, anchor_rows, anchor_cols, cluster_idx
        ),
    )
    def _get_beta_from_factors(
        self, X, missing_row, anchor_rows, anchor_cols, cluster_idx
    ):
        _anchor_rows = np.array(list(anchor_rows), dtype=int)
        _anchor_cols = np.array(list(anchor_cols), dtype=int)

        cluster = self.clusters[cluster_idx]
        u_rank = cluster["u_rank"]
        v_rank = cluster["v_rank"]
        s_rank = np.array(cluster["s_rank"])
        y1 = X[missing_row, _anchor_cols]

        rows = np.vectorize(
            cluster["rows_dict"].get,
        )(_anchor_rows)
        cols = np.vectorize(
            cluster["cols_dict"].get,
        )(_anchor_cols)
        s_rank = s_rank[:]
        u_rank = u_rank[rows, :]
        v_rank = v_rank[cols, :]

        X_als = (u_rank * s_rank) @ v_rank.T
        beta = np.linalg.lstsq(X_als.T, y1, rcond=self.min_singular_value)[0]
        train_error = self._train_error(X_als.T, y1, beta)

        return beta, u_rank[:, :].T, train_error
