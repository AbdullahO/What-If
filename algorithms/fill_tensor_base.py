from abc import ABC, abstractmethod
from typing import Any, Iterator, List, Optional, Tuple, Union
import numpy as np
from numpy import float64, int64, ndarray
import numpy.typing as np_typing
import pandas as pd
import sparse
import tensorly as tl
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

    # proerties
    @property
    def tensor_shape(self):
        return self._tensor_shape

    @tensor_shape.setter
    def tensor_shape(self, shape_):
        assert isinstance(shape_, tuple)
        self._tensor_shape = shape_

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
        units_dict = self.units_dict
        if units_dict is None:
            raise Exception("self.units_dict is None, have you called fit()?")

        time_dict = self.time_dict
        if time_dict is None:
            raise Exception("self.time_dict is None, have you called fit()?")

        actions_dict = self.actions_dict
        if actions_dict is None:
            raise Exception("self.actions_dict is None, have you called fit()?")

        # TODO: validate this
        true_intervention_assignment_matrix = self.true_intervention_assignment_matrix

        # TODO: NEED TO LOAD everything above this for prediction, along with self.get_tensor_from_factors

        # Get the units time, action, and action_time_range indices
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

        # save counts of unique units, timesteps, interventions
        self.N = N
        self.T = T
        self.I = I

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
    def _fit_transform(self, Y: np_typing.NDArray[Any]) -> np_typing.NDArray[Any]:
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

    def save(self, path):
        """save trained model"""
        raise NotImplementedError()

    def save_binary(self, path):
        """save trained model to bytes"""
        raise NotImplementedError()

    def load(self, path):
        """load model from file"""
        raise NotImplementedError()

    def load_binary(self, path):
        """load trained model from bytes"""
        raise NotImplementedError()
