"""
Alternating Least Squares (ALS) algorithm:

For more information see
https://web.stanford.edu/~rezab/classes/cme323/S15/notes/lec14.pdf
http://tensorly.org/stable/modules/generated/tensorly.decomposition.parafac.html

1. Regular ALS for pandas
2. Distributed ALS for dask
3. Iterative ALS for partial_fit / dask Iterative: using Stochastic Gradient Descent (SGD)

"""


# from algorithms.base import FillTensorBase

from typing import Optional, List

import numpy as np
import tensorly as tl
import tensorly.decomposition
from numpy import float64, int64, ndarray


# TODO: inherit from FillTensorBase?
# FillTensorBase has query, but we want to save the output separately
# Could still implement query with appropriate inputs
class AlternatingLeastSquares:
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
        self.pandas_cp_factors: Optional[List[ndarray]] = None

    def __repr__(self):
        """
        print parameters of SNN class
        """
        return str(self)

    # TODO: move this to a new base class for both FillTensorBase and WhatIFAlgorithm ?
    def __str__(self):
        field_list = []
        for (k, v) in sorted(self.__dict__.items()):
            if (v is None) or (isinstance(v, (float, int))):
                field_list.append("%s=%s" % (k, v))
            elif isinstance(v, str):
                field_list.append("%s='%s'" % (k, v))
        return "%s(%s)" % (self.__class__.__name__, ", ".join(field_list))

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
        # apply PARAFAC (ALS)
        self.cp_tensor = tl.decomposition.parafac(
            tensor, self.k_factors, n_iter_max=self.max_iterations, mask=tensor_mask, init="random"
        )
        weights, factors = self.cp_tensor
        assert np.allclose(
            np.ones(self.k_factors), weights
        ), "weights should be all ones unless parafac(normalize_factors=True)"
        self.pandas_cp_factors = factors
