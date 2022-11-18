import io
from abc import abstractmethod
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import sparse
import tensorly as tl
from numpy import ndarray

from algorithms.als import AlternatingLeastSquares as ALS
from algorithms.base import WhatIFAlgorithm


class FillTensorBase(WhatIFAlgorithm):
    """Abstract class for all whatif algorithms that uses the tensor view"""

    def __init__(self) -> None:

        self.actions_dict: Optional[dict] = None
        self.units_dict: Optional[dict] = None
        self.time_dict: Optional[dict] = None
        self.tensor_nans: Optional[sparse.COO] = None
        self.tensor_cp_factors: Optional[List[tl.CPTensor]] = None

    def check_model_for_predict(self):
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

        # convert time to datetime column
        df[time_column] = pd.to_datetime(df[time_column])

        # get tensor from df and labels
        tensor, N, I, T = self._get_tensor(
            df, unit_column, time_column, actions, metrics
        )

        # tensor to matrix
        matrix = tensor.reshape([N, I * T])

        # fill matrix
        snn_imputed_matrix = self._fit_transform(matrix)

        # reshape matrix into tensor
        tensor = snn_imputed_matrix.reshape([N, T, I])

        self.tensor_nans = sparse.COO.from_numpy(np.isnan(tensor))

        # Apply Alternating Least Squares to decompose the tensor into CP form
        als_model = ALS()

        # Modifies tensor by filling nans with zeros
        als_model.fit(tensor)

        # Only save the ALS output tensors
        self.tensor_cp_factors = als_model.cp_factors

    def query(
        self,
        units: List[int],
        time: List[str],
        metric: str,
        action: str,
        action_time_range: List[str],
    ) -> pd.DataFrame:
        """returns answer to what if query"""
        self.check_model_for_predict()
        units_dict = self.units_dict
        time_dict = self.time_dict
        actions_dict = self.actions_dict
        true_intervention_assignment_matrix = self.true_intervention_assignment_matrix

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

    def _get_tensor(
        self,
        df: pd.DataFrame,
        unit_column: str,
        time_column: str,
        actions: List[str],
        metrics: List[str],
    ) -> Tuple[ndarray, int, int, int]:
        units = df[unit_column].unique()
        N = len(units)
        timesteps = df[time_column].unique()
        T = len(timesteps)
        list_of_actions = df[actions].drop_duplicates().agg("-".join, axis=1).values
        I = len(list_of_actions)
        self.actions_dict = dict(zip(list_of_actions, np.arange(I)))

        # get tensor values
        metric_matrix_df = df.pivot(
            index=unit_column, columns=time_column, values=metrics[0]
        )
        self.units_dict = dict(zip(metric_matrix_df.index, np.arange(N)))
        self.time_dict = dict(zip(metric_matrix_df.columns, np.arange(T)))

        # add the action_idx to each row
        df["intervention_assignment"] = (
            df[actions].agg("-".join, axis=1).map(self.actions_dict).values
        )

        # create assignment matrix
        self.true_intervention_assignment_matrix = df.pivot(
            index=unit_column, columns=time_column, values="intervention_assignment"
        ).values

        # init tensor
        tensor = np.full([N, T, I], np.nan)

        # fill tensor with metric_matrix values in appropriate locations
        metric_matrix = metric_matrix_df.values
        for action_idx in range(I):
            unit_time_received_action_idx = (
                self.true_intervention_assignment_matrix == action_idx
            )
            tensor[unit_time_received_action_idx, action_idx] = metric_matrix[
                unit_time_received_action_idx
            ]

        return tensor, N, I, T

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
        self.check_model_for_predict()
        tensor_nans_dict = self.sparse_COO_to_dict(self.tensor_nans)
        model_dict = {
            "actions_dict": self.actions_dict,
            "units_dict": self.units_dict,
            "time_dict": self.time_dict,
            "tensor_cp_factors": np.array(self.tensor_cp_factors),
            "true_intervention_assignment_matrix": self.true_intervention_assignment_matrix,
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
        self.tensor_nans = tensor_nans
        self.check_model_for_predict()

    def load_binary(self, bytes_str):
        """
        Load trained model from bytes
        """
        # Allow pickle is required to save/load dictionaries.
        # We could save them separately instead.
        bytes_io = io.BytesIO(bytes_str)
        self.load(bytes_io)
