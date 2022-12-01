import io
import warnings
from abc import abstractmethod
from collections import namedtuple
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import sparse
import tensorly as tl
from numpy import ndarray
from sklearn.utils import check_array  # type: ignore

from algorithms.als import AlternatingLeastSquares as ALS
from algorithms.base import WhatIFAlgorithm

ModelTuple = namedtuple(
    "ModelTuple",
    "actions_dict units_dict time_dict tensor_nans tensor_cp_factors true_intervention_assignment_matrix unit_column time_column actions metric",
)


class FillTensorBase(WhatIFAlgorithm):
    """Base class for all whatif algorithms that uses the tensor view"""

    def __init__(
        self, verbose: Optional[bool] = False, min_singular_value: float = 1e-7
    ) -> None:

        self.verbose = verbose
        self.min_singular_value = min_singular_value
        self.actions_dict: Optional[dict] = None
        self.units_dict: Optional[dict] = None
        self.time_dict: Optional[dict] = None
        self.tensor_nans: Optional[sparse.COO] = None
        self.tensor_cp_factors: Optional[List[tl.CPTensor]] = None
        self.metric: Optional[str] = None
        self.unit_column: Optional[str] = None
        self.time_column: Optional[str] = None
        self.actions: Optional[List[str]] = None
        self.covariates: Optional[List[str]] = None
        self.true_intervention_assignment_matrix: Optional[ndarray] = None

    def check_model(self):
        error_message_base = " is None, have you called fit()?"
        if self.tensor_cp_factors is None:
            raise ValueError(f"self.tensor_cp_factors{error_message_base}")
        # TODO: right now we will raise ValueError if tensor_nans is None, should we allow None?
        if self.tensor_nans is None:
            raise ValueError(f"self.tensor_nans{error_message_base}")
        if self.units_dict is None:
            raise ValueError(f"self.units_dict{error_message_base}")
        if self.time_dict is None:
            raise ValueError(f"self.time_dict{error_message_base}")
        if self.actions_dict is None:
            raise ValueError(f"self.actions_dict{error_message_base}")
        if self.true_intervention_assignment_matrix is None:
            raise ValueError(
                f"self.true_intervention_assignment_matrix{error_message_base}"
            )
        if self.unit_column is None:
            raise ValueError(f"self.unit_column{error_message_base}")
        if self.time_column is None:
            raise ValueError(f"self.time_column{error_message_base}")
        if self.actions is None:
            raise ValueError(f"self.actions{error_message_base}")
        if self.metric is None:
            raise ValueError(f"self.metric{error_message_base}")

        return ModelTuple(
            self.actions_dict,
            self.units_dict,
            self.time_dict,
            self.tensor_nans,
            self.tensor_cp_factors,
            self.true_intervention_assignment_matrix,
            self.unit_column,
            self.time_column,
            self.actions,
            self.metric,
        )

    @property
    def T(self):
        if self.time_dict:
            _T = len(self.time_dict.values())
            return _T
        else:
            return None

    @property
    def I(self):
        if self.actions_dict:
            _I = len(self.actions_dict.values())
            return _I
        else:
            return None

    @property
    def N(self):
        if self.units_dict:
            _N = len(self.units_dict.values())
            return _N
        else:
            return None

    def fit(
        self,
        df: pd.DataFrame,
        unit_column: str,
        time_column: str,
        metrics: list,
        actions: list,
        covariates: Optional[list] = None,
    ) -> None:
        """take sparse tensor and fill it!

        Args:
            df (pd.DataFrame): data dataframe
            unit_column (str): name for the unit column
            time_column (str): name for the time column
            metrics (list): list of names for the metric columns
            actions (list): list of names for the action columns
            covariates (list, optional): list of names for the covariate columns. Defaults to None.

        """
        assert len(metrics) == 1, "method can only support single metric for now"
        self.metric = metrics[0]
        self.unit_column = unit_column
        self.time_column = time_column
        self.actions = actions
        self.covariates = covariates

        # get tensor from df and labels
        tensor = self._get_tensor(df, unit_column, time_column, actions, metrics)

        # fill tensor
        tensor_filled = self._fit_transform(tensor)

        self.tensor_nans = sparse.COO.from_numpy(np.isnan(tensor_filled))

        # Apply Alternating Least Squares to decompose the tensor into CP form
        als_model = ALS()

        # Modifies tensor by filling nans with zeros
        als_model.fit(tensor_filled)

        # Only save the ALS output tensors
        self.tensor_cp_factors = als_model.cp_factors
        assert (
            self.tensor_cp_factors is not None
        ), "self.tensor_cp_factors is None, check ALS model fit"
        self.tensor_cp_combined_action_unit_factors = self._merge_factors(
            self.tensor_cp_factors[0], self.tensor_cp_factors[2]
        )

    def query(
        self,
        units: List[int],
        time: List[str],
        metric: str,
        action: str,
        action_time_range: List[str],
    ) -> pd.DataFrame:
        """returns answer to what if query"""
        model_tuple = self.check_model()
        units_dict = model_tuple.units_dict
        time_dict = model_tuple.time_dict
        actions_dict = model_tuple.actions_dict
        true_intervention_assignment_matrix = (
            model_tuple.true_intervention_assignment_matrix
        )

        # Get the units, time, action, and action_time_range indices
        unit_idx = [units_dict[u] for u in units]
        # convert to timestamp
        _time = [pd.Timestamp(t) for t in time]
        # get all timesteps in range
        timesteps = [t for t in time_dict.keys() if t <= _time[1] and t >= _time[0]]
        # get idx
        time_idx = [time_dict[t] for t in timesteps]

        tensor = self.get_tensor_from_factors(unit_idx=unit_idx, time_idx=time_idx)

        # convert to timestamp
        _action_time_range = [pd.Timestamp(t) for t in action_time_range]
        # get all timesteps in action range
        action_timesteps = [
            t
            for t in time_dict.keys()
            if t <= _action_time_range[1] and t >= _action_time_range[0]
        ]
        # get idx
        action_time_idx = [time_dict[t] for t in action_timesteps]
        # get action idx
        action_idx = actions_dict[action]
        # Get default actions assignment for units of interest
        assignment = np.array(true_intervention_assignment_matrix[unit_idx, :])
        # change assignment according to the requested action range
        assignment[:, action_time_idx] = action_idx
        # get it for only the time frame of interest
        assignment = assignment[:, time_idx]

        # Select the right matrix based on the actions selected
        assignment = assignment.reshape([assignment.shape[0], assignment.shape[1], 1])
        selected_matrix = np.take_along_axis(tensor, assignment, axis=2)[:, :, 0]

        # return unit X time DF
        out_df = pd.DataFrame(data=selected_matrix, index=units, columns=timesteps)
        return out_df

    def _get_metric_matrix(
        self, df: pd.DataFrame, unit_column: str, time_column: str, metric: str
    ) -> pd.DataFrame:
        metric_matrix_df = df.pivot(
            index=unit_column, columns=time_column, values=metric
        )
        return metric_matrix_df

    def _get_assignment_matrix(
        self, df: pd.DataFrame, unit_column: str, time_column: str, actions: List[str]
    ) -> ndarray:
        # add the action_idx to each row
        df["intervention_assignment"] = (
            df[actions].agg("-".join, axis=1).map(self.actions_dict).values
        )

        # create assignment matrix
        current_true_intervention_assignment_matrix = df.pivot(
            index=unit_column,
            columns=time_column,
            values="intervention_assignment",
        ).values
        return current_true_intervention_assignment_matrix

    def _process_input_df(
        self,
        df: pd.DataFrame,
        unit_column: str,
        time_column: str,
        metric: str,
        actions: List[str],
    ) -> Tuple[pd.DataFrame, ndarray]:
        # convert time to datetime column
        df[time_column] = pd.to_datetime(df[time_column])

        # get metric values in a (unit x time) matrix
        metric_matrix_df = self._get_metric_matrix(df, unit_column, time_column, metric)

        # get assignment (unit x time) matrix
        current_true_intervention_assignment_matrix = self._get_assignment_matrix(
            df, unit_column, time_column, actions
        )
        return metric_matrix_df, current_true_intervention_assignment_matrix

    def _get_partial_tensor(
        self,
        df: pd.DataFrame,
    ) -> ndarray:
        model_tuple = self.check_model()
        unit_column = model_tuple.unit_column
        time_column = model_tuple.time_column
        metric = model_tuple.metric
        actions = model_tuple.actions
        time_dict = model_tuple.time_dict

        (
            metric_matrix_df,
            current_true_intervention_assignment_matrix,
        ) = self._process_input_df(df, unit_column, time_column, metric, actions)
        timesteps = df[time_column].unique()
        T = len(timesteps)

        max_t = max(list(time_dict.values()))
        time_dict.update(
            dict(zip(metric_matrix_df.columns, np.arange(max_t, max_t + T)))
        )

        ## TODO: potential storage issue, we're storing an N xT matrix that grows with T (and N).
        self.true_intervention_assignment_matrix = np.concatenate(
            [
                self.true_intervention_assignment_matrix,
                current_true_intervention_assignment_matrix,
            ],
            axis=1,
        )

        tensor = self._populate_tensor(
            self.N,
            T,
            self.I,
            metric_matrix_df,
            current_true_intervention_assignment_matrix,
        )
        return tensor

    def _get_tensor(
        self,
        df: pd.DataFrame,
        unit_column: str,
        time_column: str,
        actions: List[str],
        metrics: List[str],
    ) -> ndarray:

        # populate actions dict
        list_of_actions = df[actions].drop_duplicates().agg("-".join, axis=1).values
        I = len(list_of_actions)
        self.actions_dict = dict(zip(list_of_actions, np.arange(I)))

        (
            metric_matrix_df,
            current_true_intervention_assignment_matrix,
        ) = self._process_input_df(df, unit_column, time_column, metrics[0], actions)

        # populate units dict
        units = df[unit_column].unique()
        N = len(units)
        self.units_dict = dict(zip(metric_matrix_df.index, np.arange(N)))

        # populate time dict
        timesteps = df[time_column].unique()
        T = len(timesteps)
        self.time_dict = dict(zip(metric_matrix_df.columns, np.arange(T)))

        self.true_intervention_assignment_matrix = (
            current_true_intervention_assignment_matrix
        )
        tensor = self._populate_tensor(
            self.N,
            T,
            self.I,
            metric_matrix_df,
            current_true_intervention_assignment_matrix,
        )

        return tensor

    @staticmethod
    def _populate_tensor(N, T, I, metric_df, assignment_matrix):
        # init tensor
        tensor = np.full([N, T, I], np.nan)

        # fill tensor with metric_matrix values in appropriate locations
        metric_matrix = metric_df.values
        for action_idx in range(I):
            unit_time_received_action_idx = assignment_matrix == action_idx
            tensor[unit_time_received_action_idx, action_idx] = metric_matrix[
                unit_time_received_action_idx
            ]
        return tensor

    def get_tensor_from_factors(
        self,
        unit_idx: Optional[List[int]] = None,
        time_idx: Optional[List[int]] = None,
    ):
        tensor = ALS._predict(
            self.tensor_cp_factors, unit_idx=unit_idx, time_idx=time_idx
        )
        if self.tensor_nans is not None:
            # Assumes factors are in order N, T, I (unit, time, intervention)
            tensor_nans = self.tensor_nans
            if unit_idx is not None:
                tensor_nans = tensor_nans[unit_idx]
            if time_idx is not None:
                tensor_nans = tensor_nans[:, time_idx]
            mask = tensor_nans.todense()
            tensor[mask] = np.nan
        return tensor

    def partial_fit(
        self,
        new_df: pd.DataFrame,
    ) -> None:

        new_tensor = self._get_partial_tensor(new_df)
        new_time_factors = self.update_time_factors(new_tensor)
        # update nan mask
        ## TODO: update nan mask based on original fit:
        #        1. if intervention/uni/time never observed --> nan (for all methods)
        #        2. if (time,intervention) never observed --> nan (for SNN)
        #        3. if (time,unit) never observed --> nan (for SNN)
        assert (
            self.tensor_nans is not None
        ), "self.tensor_nans is None, have you called fit()?"
        old_shape = self.tensor_nans.shape
        self.tensor_nans.shape = (
            old_shape[0],
            new_time_factors.shape[0],
            old_shape[2],
        )

    def update_time_factors(self, new_tensor: ndarray) -> ndarray:
        """update time factors using the new observations in new tensor ([N x new_timesteps x I])

        Args:
            new_tensor (ndarray): new tensor that correspond to new obseravations

        Returns:
            ndarray: new time factors (new_timesteps x k)
        """
        # check if self.tensor_cp_factors is not None
        assert (
            self.tensor_cp_factors is not None
        ), "self.tensor_cp_factors, have you called fit()?"

        # transfrom tensor to matrix (NI X T)
        N, T, I = new_tensor.shape
        matrix = new_tensor.transpose([2, 0, 1]).reshape([N * I, T])

        # update factors based on new columns
        Y = self.tensor_cp_factors[1]
        X = self.tensor_cp_combined_action_unit_factors
        k = Y.shape[1]
        Y_new = np.zeros([Y.shape[0] + matrix.shape[1], k])
        Y_new[: Y.shape[0], :] = Y
        for col_i in range(matrix.shape[1]):
            col = matrix[:, col_i]
            observed_entries = np.argwhere(~np.isnan(col)).flatten().astype(int)
            mat = X[observed_entries, :].T @ X[observed_entries, :]
            b = X[observed_entries, :].T @ col[observed_entries]
            assert b.shape == (k,), b.shape
            assert mat.shape == (k, k), mat.shape
            Y_new[Y.shape[0] + col_i, :] = np.linalg.lstsq(
                mat, b, rcond=self.min_singular_value
            )[0]

        self.tensor_cp_factors[1] = Y_new
        return Y_new

    @abstractmethod
    def _fit_transform(self, Y: ndarray) -> ndarray:
        """take sparse tensor and return a full tensor

        Args:
            Y (np.array): sparse tensor

        Returns:
            np.array: filled tensor
        """

    def diagnostics(self):
        """returns method-specifc diagnostics"""
        raise NotImplementedError()

    def summary(self):
        """returns method-specifc summary"""
        raise NotImplementedError()

    @staticmethod
    def sparse_COO_to_dict(sparse_array: sparse.COO):
        # tensor_nans mask is saved as a sparse.COO array
        # Get parts of sparse array to save individually
        return {
            "data": sparse_array.data,
            "shape": sparse_array.shape,
            "fill_value": sparse_array.fill_value,
            "coords": sparse_array.coords,
        }

    @staticmethod
    def sparse_COO_from_dict(sparse_array_dict: dict) -> sparse.COO:
        # https://github.com/pydata/sparse/blob/master/sparse/_io.py#L71
        coords = sparse_array_dict["coords"]
        data = sparse_array_dict["data"]
        shape = tuple(sparse_array_dict["shape"])
        fill_value = sparse_array_dict["fill_value"][()]
        return sparse.COO(
            coords=coords,
            data=data,
            shape=shape,
            sorted=True,
            has_duplicates=False,
            fill_value=fill_value,
        )

    def get_model_dict_for_save(self):
        model_tuple = self.check_model()
        tensor_nans_dict = self.sparse_COO_to_dict(model_tuple.tensor_nans)
        model_dict = {
            "actions_dict": model_tuple.actions_dict,
            "units_dict": model_tuple.units_dict,
            "time_dict": model_tuple.time_dict,
            "tensor_cp_factors": np.array(
                model_tuple.tensor_cp_factors, dtype="object"
            ),
            "true_intervention_assignment_matrix": model_tuple.true_intervention_assignment_matrix,
            "unit_column": model_tuple.unit_column,
            "time_column": model_tuple.time_column,
            "actions": model_tuple.actions,
            "metric": model_tuple.metric,
            **{f"tensor_nans_{key}": value for key, value in tensor_nans_dict.items()},
        }
        return model_dict

    def save(self, path_or_obj):
        """
        Save trained model to a file
        """
        # TODO: is there a way to save to a representation where we don't have to load the whole factor matrix?
        model_dict = self.get_model_dict_for_save()
        np.savez_compressed(path_or_obj, **model_dict)

    def save_binary(self):
        """
        Save trained model to a byte string
        """
        bytes_io = io.BytesIO()
        self.save(bytes_io)
        bytes_str = bytes_io.getvalue()
        return bytes_str

    def load(self, path_or_obj):
        """
        Load trained model from file
        """
        loaded_dict = dict(np.load(path_or_obj, allow_pickle=True))
        actions_dict = loaded_dict.pop("actions_dict").tolist()
        units_dict = loaded_dict.pop("units_dict").tolist()
        time_dict = loaded_dict.pop("time_dict").tolist()
        tensor_cp_factors = loaded_dict.pop("tensor_cp_factors").tolist()
        true_intervention_assignment_matrix = loaded_dict.pop(
            "true_intervention_assignment_matrix"
        )
        unit_column = loaded_dict.pop("unit_column")
        time_column = loaded_dict.pop("time_column")
        actions = loaded_dict.pop("actions").tolist()
        metric = loaded_dict.pop("metric")

        tensor_nans_prefix = "tensor_nans_"
        tensor_nans_dict = {
            key[len(tensor_nans_prefix) :]: value
            for key, value in loaded_dict.items()
            if key.startswith(tensor_nans_prefix)
        }
        tensor_nans = self.sparse_COO_from_dict(tensor_nans_dict)
        self.actions_dict = actions_dict
        self.units_dict = units_dict
        self.time_dict = time_dict
        self.tensor_cp_factors = tensor_cp_factors
        self.true_intervention_assignment_matrix = true_intervention_assignment_matrix
        self.unit_column = unit_column
        self.time_column = time_column
        self.actions = actions
        self.metric = metric
        self.tensor_nans = tensor_nans
        self.check_model()

    def load_binary(self, bytes_str):
        """
        Load trained model from bytes
        """
        # Allow pickle is required to save/load dictionaries.
        # We could save them separately instead.
        bytes_io = io.BytesIO(bytes_str)
        self.load(bytes_io)

    def _check_input_matrix(self, X: ndarray, missing_mask: ndarray, ndim: int) -> None:
        """
        check to make sure that the input matrix
        and its mask of missing values are valid.
        """
        if len(X.shape) != ndim:
            raise ValueError(
                "expected %dd matrix, got %s array"
                % (
                    ndim,
                    X.shape,
                )
            )
        if not len(missing_mask) > 0:
            warnings.simplefilter("always")
            warnings.warn("input matrix is not missing any values")
        if len(missing_mask) == int(np.prod(X.shape)):
            raise ValueError(
                "input matrix must have some observed (i.e., non-missing) values"
            )

    def _prepare_input_data(
        self, X: ndarray, missing_mask: ndarray, ndim: int
    ) -> ndarray:
        """
        prepare input matrix X. return if valid else terminate
        """
        X = check_array(X, force_all_finite=False, allow_nd=True)
        if (X.dtype != "f") and (X.dtype != "d"):
            X = X.astype(float)
        self._check_input_matrix(X, missing_mask, ndim)
        return X

    def _merge_factors(self, X, Y):
        assert X.shape[1] == Y.shape[1]
        k = X.shape[1]
        n = Y.shape[0]
        m = X.shape[0]
        vw = np.zeros([m * n, k])
        for i in range(n):
            vw[i * m : (i + 1) * m, :] = Y[i, :] * X
        return vw
