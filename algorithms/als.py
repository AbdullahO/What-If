"""
Alternating Least Squares (ALS) algorithm:

For more information see
https://web.stanford.edu/~rezab/classes/cme323/S15/notes/lec14.pdf
http://tensorly.org/stable/modules/generated/tensorly.decomposition.parafac.html

1. Regular ALS for pandas
2. Distributed ALS for dask
<s>3. Iterative ALS for partial_fit / dask Iterative: using Stochastic Gradient Descent (SGD)</s>
Abdullah's idea for Iterative ALS: only update T factor
    - keep window of T available for predictions
    - need to allow user to change the window and/or automatically determine
"""


from typing import List, Optional

import numpy as np
import tensorly as tl
import tensorly.decomposition
from numpy import float64, int64, ndarray

from algorithms.base import StrReprBase


class AlternatingLeastSquares(StrReprBase):
    """
    Impute missing entries in a matrix using the ALS algorithm
    """

    def __init__(self, max_iterations: int = 1000, k_factors: int = 5) -> None:
        """
        Parameters
        ----------
        ???

        """

        self.max_iterations = max_iterations
        self.k_factors = k_factors
        self.cp_tensor: Optional[tl.CPTensor] = None
        self.cp_factors: Optional[List[ndarray]] = None

    def fit(self, tensor: ndarray) -> None:
        """
        Use Tensorly Parafac to apply ALS to a numpy array tensor that fits in memory (Pandas route)

        We use self.k_factors for what Tensorly calls rank. The output will be a CPTensor which is a tuple of
        (weights, factors) where factors is a list of factor matrices of shape (tensor.shape[i], self.k_factors),
        with a factor matrix for each axis of the input tensor.
        """
        # create mask of ones and zeros, with zeros for nan
        tensor_mask = np.ones_like(tensor)
        tensor_mask[np.isnan(tensor)] = 0
        # fill zeros for nan
        tensor[np.isnan(tensor)] = 0
        N, T, I = tensor.shape
        # apply PARAFAC (ALS)
        np.random.seed(0)
        try:
            self.cp_tensor = tl.decomposition.parafac(
                tensor,
                self.k_factors,
                n_iter_max=self.max_iterations,
                mask=tensor_mask,
            )
        ## to resolve singular matrices issue by adding some gausian noise
        except:
            print("warning, the approximation may not be perfect due to noise added for stability!")
            tensor_adjusted = tensor + 1e-2 * tensor.mean() * np.random.randn(N, T, I)
            self.cp_tensor = tl.decomposition.parafac(
                tensor_adjusted,
                self.k_factors,
                n_iter_max=self.max_iterations,
                mask=tensor_mask,
            )
        weights, factors = self.cp_tensor
        assert np.allclose(
            np.ones(self.k_factors), weights
        ), "weights should be all ones unless parafac(normalize_factors=True)"
        self.cp_factors = factors

    @staticmethod
    def _predict(
        factors: Optional[List[ndarray]] = None,
        unit_idx: Optional[List[int]] = None,
        time_idx: Optional[List[int]] = None,
    ) -> ndarray:
        if not factors:
            error_message = "factors not passed to _predict, did you mean to use the instance method: predict?"
            raise ValueError(error_message)
        factors = factors.copy()
        rank = factors[0].shape[1]
        # Assumes factors are in order N, T, I (unit, time, intervention)
        if unit_idx is not None:
            factors[0] = factors[0][unit_idx]
        if time_idx is not None:
            factors[1] = factors[1][time_idx]
        cp_tensor = (np.ones(rank), factors)
        full_tensor = tl.cp_to_tensor(cp_tensor)
        return full_tensor

    def predict(
        self,
        unit_idx: Optional[List[int]] = None,
        time_idx: Optional[List[int]] = None,
    ) -> ndarray:
        factors = self.cp_factors
        if factors is None:
            error_message = "self.cp_factors is None: have you called fit()?"
            raise ValueError(error_message)
        full_tensor = AlternatingLeastSquares._predict(factors, unit_idx, time_idx)
        return full_tensor
