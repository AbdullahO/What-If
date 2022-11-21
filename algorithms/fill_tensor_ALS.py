"""
ALS for what IF
"""
import warnings
from typing import Any, Iterator, List, Optional, Tuple, Union
import tensorly as tl
import numpy as np
from numpy import float64, int64, ndarray
from algorithms.fill_tensor_base import FillTensorBase


class ALS(FillTensorBase):
    """
    Impute missing entries in a matrix via SNN algorithm + biclustering
    """

    def __init__(
        self,
        init: str = "random",
        ranks: List[int] = [2, 3, 5, 7],
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        validation_split: Optional[float] = 0.1,
        verbose: Optional[bool] = True,
        min_singular_value: float = 1e-7,
    ):
        """
        Parameters
        ----------

        rank : int
        Perform ALS on training data with this value as its rank

        init : str
        how to initialize the factors in ALS, choose from {"random", "svd"}

        min_value : float
        Minumum possible imputed value

        max_value : float
        Maximum possible imputed value

        verbose : bool

        validation_split : float
        fraction of data used for validation
        """
        super().__init__(verbose=verbose, min_singular_value=min_singular_value)

        self.ranks = ranks
        self.init = init
        self.min_value = min_value
        self.max_value = max_value
        self.tensor = None
        self.mask = None
        self.actions_dict = None
        self.units_dict = None
        self.time_dict = None
        self.feasible = None
        self.validation_split = validation_split

    def _estimate_rank(self, tensor):
        self.mask = (~np.isnan(tensor)).astype(int)
        observed_set = np.argwhere(~np.isnan(tensor))
        num_validation_points = int(len(observed_set) * self.validation_split)
        validation_set = np.random.choice(
            np.arange(len(observed_set)), num_validation_points
        )
        artificial_mask = np.array(self.mask)
        validation_index = observed_set[validation_set]
        artificial_mask[
            validation_index[:, 0], validation_index[:, 1], validation_index[:, 2]
        ] = 0
        X = np.array(tensor)
        X[artificial_mask == 0] = np.nan
        min_error = np.inf
        # selected_rank = 1
        for rank in self.ranks:
            X_ = self.prepare_input_data(X, missing_set=[(0, 0)])
            # initialize
            weights, factors = tl.decomposition.parafac(
                X_,
                rank=rank,
                mask=artificial_mask,
                init=self.init,
                normalize_factors=True,
            )
            tensor_imputed = tl.cp_to_tensor((weights, factors))
            tensor_imputed = self._clip(tensor_imputed)
            error = np.nanmean(
                np.square(
                    tensor[artificial_mask == 0] - tensor_imputed[artificial_mask == 0]
                )
            )
            if error < min_error:
                selected_rank = rank
                min_error = error
        if self.verbose:
            print(f"selected rank is {selected_rank}")
        return selected_rank

    def _fit_transform(self, tensor):
        """
        complete missing entries in matrix
        """
        rank = self._estimate_rank(tensor)
        # get missing entries to impute
        missing_set = np.argwhere(np.isnan(tensor))
        # check and prepare data
        X = self.prepare_input_data(tensor, missing_set)

        # initialize
        weights, factors = tl.decomposition.parafac(
            X,
            rank=rank,
            mask=self.mask,
            init=self.init,
            normalize_factors=True,
        )
        tensor_imputed = tl.cp_to_tensor((weights, factors))
        tensor_imputed = self._clip(tensor_imputed)
        self.feasible = np.ones_like(tensor_imputed)
        return tensor_imputed

    def prepare_input_data(self, tensor, missing_set):
        tensor = self._prepare_input_data(tensor, missing_set, 3)
        tensor[np.isnan(tensor)] = 0
        return tensor

    def _clip(self, x):
        """
        clip values to fall within range [min_value, max_value]
        """
        if self.min_value is not None:
            x[x < self.min_value] = self.min_value
        if self.max_value is not None:
            x[x > self.max_value] = self.max_value
        return x
