import io
import warnings
from abc import abstractmethod
from collections import namedtuple
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import sparse
from numpy import ndarray
from sklearn.utils import check_array  # type: ignore

from .als import AlternatingLeastSquares as ALS
from .base import WhatIFAlgorithm
from .mssa import MSSA

ModelTuple = namedtuple(
    "ModelTuple",
    "actions_dict units_dict time_dict tensor_nans regimes true_intervention_assignment_matrix unit_column time_column actions metric current_regime_tensor cusum distance_error",
)


class Regime(object):
    # This may be changed to a namedTuple or dataclass?
    """Base class for data generation regimes, each regime will consist of its own latent factors"""

    def __init__(self, index, start_time):
        self.index: int = index
        self.start_time: int = start_time
        self.end_time: Optional[int] = None
        self.tensor_cp_factors: Optional[ndarray] = None
        self.tensor_cp_combined_action_unit_factors: Optional[ndarray] = None
        self.mean_drift: Optional[float] = None
        self.forecasting_model: List[MSSA] = None


class FillTensorBase(WhatIFAlgorithm):
    """Base class for all whatif algorithms that uses the tensor view"""

    def __init__(
        self,
        verbose: Optional[bool] = False,
        min_singular_value: float = 1e-7,
        full_training_time_steps: int = 20,
        threshold_multiplier: int = 10,
        L: Optional[int] = None,
        k_factors: Optional[int] = 5,
        num_lags_forecasting: int = 10,
    ) -> None:
        self.verbose = verbose
        self.L = L
        self.num_lags_forecasting = num_lags_forecasting
        self.min_singular_value = min_singular_value
        self.actions_dict: Optional[dict] = None
        self.units_dict: Optional[dict] = None
        self.time_dict: Optional[dict] = None
        self.tensor_nans: Optional[sparse.COO] = None
        self.metric: Optional[str] = None
        self.unit_column: Optional[str] = None
        self.time_column: Optional[str] = None
        self.actions: Optional[List[str]] = None
        self.covariates: Optional[List[str]] = None
        self.true_intervention_assignment_matrix: Optional[ndarray] = None
        self.distance_error: Optional[ndarray] = None
        self.cusum: Optional[ndarray] = None
        self.full_training_time_steps = full_training_time_steps
        self.current_regime_tensor: Optional[ndarray] = None
        self.threshold_multiplier: int = threshold_multiplier
        self.regimes: List[Regime] = []
        # init initial regime!
        strating_regime = Regime(0, 0)
        self.regimes.append(strating_regime)
        self.k_factors: int = k_factors

    def check_model(self):
        error_message_base = " is None, have you called fit()?"
        if len(self.regimes) == 0:
            raise ValueError(f"self.regime {error_message_base}")
        for r, regime in enumerate(self.regimes):
            if regime.tensor_cp_factors is None:
                raise ValueError(
                    f"for regime {r}, self.tensor_cp_factors{error_message_base}"
                )
            if regime.tensor_cp_combined_action_unit_factors is None:
                raise ValueError(
                    f"for regime {r}, self.tensor_cp_combined_action_unit_factors{error_message_base}"
                )
            if regime.mean_drift is None:
                raise ValueError(f"for regime {r}, self.mean_drift{error_message_base}")
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
        if self.cusum is None:
            raise ValueError(f"self.cusum{error_message_base}")
        if self.distance_error is None:
            raise ValueError(f"self.distance_error{error_message_base}")

        return ModelTuple(
            self.actions_dict,
            self.units_dict,
            self.time_dict,
            self.tensor_nans,
            self.regimes,
            self.true_intervention_assignment_matrix,
            self.unit_column,
            self.time_column,
            self.actions,
            self.metric,
            self.current_regime_tensor,
            self.cusum,
            self.distance_error,
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
        tensor = self._get_tensor(
            df, self.unit_column, self.time_column, self.actions, [self.metric]
        )

        # init initial regime
        self._fit(tensor, 0)

    def _fit(self, tensor, regime_index):
        # get regime
        regime = self.regimes[regime_index]

        # fill tensor
        N, T, I = tensor.shape
        X = self._tensor_to_matrix(tensor, N, T, I)

        filled_matrix = self._fit_transform(X)
        tensor_filled = self._matrix_to_tensor(filled_matrix, N, T, I)

        self.tensor_nans = self._update_tensor_nans(tensor_filled, regime)

        # Apply Alternating Least Squares to decompose the tensor into CP form
        regime.tensor_cp_factors = self._compress_tensor(tensor_filled)

        # calculate mean error to detect drift
        tensor_compressed = self.get_tensor_from_factors(regime.tensor_cp_factors)
        regime.mean_drift = (
            2
            * np.nanmean(np.square(tensor_compressed - tensor))
            / np.nanmean(np.square(tensor))
        )

        # necessary post processing needed for partial fit
        assert (
            regime.tensor_cp_factors is not None
        ), "self.tensor_cp_factors is None, check ALS model fit"
        regime.tensor_cp_combined_action_unit_factors = self._merge_factors(
            regime.tensor_cp_factors[0], regime.tensor_cp_factors[2]
        )

        if tensor.shape[1] < self.full_training_time_steps:
            # store the original tensor for potential full update
            self.current_regime_tensor = np.array(tensor)

        # init distance metric for drift
        if self.distance_error is None:
            self.distance_error = np.zeros(tensor.shape[1])
            self.cusum = np.zeros(tensor.shape[1])
        else:
            self.distance_error = self.distance_error[: regime.start_time]
            self.distance_error = np.concatenate(
                [self.distance_error, np.zeros(tensor.shape[1])]
            )
            self.cusum = self.cusum[: regime.start_time]
            self.cusum = np.concatenate([self.cusum, np.zeros(tensor.shape[1])])

        # forecasting time factors
        self._fit_forecasting(regime)

    def _fit_forecasting(self, regime):
        # temporal factors ar at index 1!
        temporal_factors = regime.tensor_cp_factors[1]
        assert temporal_factors.shape == (
            self.T,
            self.k_factors,
        ), temporal_factors.shape

        # init models
        kargs = {"numSeries": 1, "numCoefs": self.num_lags_forecasting, "arOrder": 0}
        regime.forecasting_model = [MSSA(**kargs) for _ in range(self.k_factors)]
        for factor in range(self.k_factors):
            regime.forecasting_model[factor].fit(
                temporal_factors[:, factor : factor + 1]
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

        tensor = self.query_tensor(unit_idx=unit_idx, time_idx=time_idx)

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

    def query_tensor(self, unit_idx, time_idx):
        # find relevant regimes based on time
        start_idx, end_idx = time_idx[0], time_idx[-1]
        # filter based on end_idx
        regimes = [regime for regime in self.regimes if regime.start_time <= end_idx]
        # filter based on start_idx, regime.end_time may be None; when it is Nonr it should pass
        regimes = [
            regime
            for regime in regimes
            if float(regime.end_time or start_idx) >= start_idx
        ]

        # query from each regime
        tensor = np.zeros([len(unit_idx), 0, self.I])
        for regime in regimes:
            regime_time_idx_start = (
                max(start_idx, regime.start_time) - regime.start_time
            )
            regime_time_idx_end = (
                min(end_idx, float(regime.end_time or end_idx)) - regime.start_time
            )
            regime_time_idx = np.arange(
                regime_time_idx_start, regime_time_idx_end + 1
            ).astype(int)
            tensor_regime = self.get_tensor_from_factors(
                regime.tensor_cp_factors, unit_idx=unit_idx, time_idx=regime_time_idx
            )
            tensor = np.concatenate([tensor, tensor_regime], axis=1)

        # mask and then return tensor
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

    def forecast(
        self,
        units: List[int],
        steps_ahead: int,
        metric: str,
        action: str,
    ) -> pd.DataFrame:
        """returns answer to what if query"""
        # checo model
        model_tuple = self.check_model()
        units_dict = model_tuple.units_dict
        actions_dict = model_tuple.actions_dict

        # Get the units, time, action, and action_time_range indices
        unit_idx = [units_dict[u] for u in units]

        tensor = self._forecast_tensor(unit_idx=unit_idx, steps_ahead=steps_ahead)

        # get action idx
        action_idx = actions_dict[action]

        selected_matrix = tensor[:, :, action_idx]

        # return unit X time DF
        # for now, we just return how the index of the steps and not the timestamps, for which we will need to
        # learn the frequency of the observations and extrapolate it
        timesteps = np.arange(self.T, self.T + steps_ahead)
        out_df = pd.DataFrame(data=selected_matrix, index=units, columns=timesteps)
        return out_df

    def _forecast_tensor(self, unit_idx, steps_ahead):
        # regime is "always" the latest one when we forecast
        regime = self.regimes[-1]

        # step one: forecast factors (get steps_ahead X k_factors x )
        temporal_factors_forecasted = np.zeros([steps_ahead, self.k_factors])
        for k in range(self.k_factors):
            temporal_factors_forecasted[:, k : k + 1] = regime.forecasting_model[
                k
            ].predict(numSteps=steps_ahead)
        # step two: get the right tensor for regime and use get_tensor_from_factors
        factors = [
            regime.tensor_cp_factors[0],
            temporal_factors_forecasted,
            regime.tensor_cp_factors[2],
        ]
        tensor_forecasted = self.get_tensor_from_factors(factors, unit_idx=unit_idx)

        return tensor_forecasted

    def _update_tensor_nans(self, tensor_filled, regime):
        tensor_nans = sparse.COO.from_numpy(np.isnan(tensor_filled))
        # new tensor_nan !
        if self.tensor_nans is None:
            return tensor_nans
        # append to old tensor based on regime start time
        else:
            regime_start = regime.start_time
            old_tensor_coords = self.tensor_nans.coords
            old_shape = (
                self.tensor_nans.shape[0],
                regime_start,
                self.tensor_nans.shape[2],
            )
            indices = [
                i for i, t in enumerate(old_tensor_coords[1]) if t < regime_start
            ]
            data = self.tensor_nans.data[indices]
            coords = [
                self.tensor_nans.coords[0][indices],
                self.tensor_nans.coords[1][indices],
                self.tensor_nans.coords[2][indices],
            ]
            old_tensor = sparse.COO(
                coords=coords, data=data, shape=old_shape, sorted=True
            )
            return sparse.concatenate([old_tensor, tensor_nans], axis=1)

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

    def _tensor_to_matrix(self, X, N, T, I):
        if self.L:
            return self.pagify(X, self.L, N, T, I)
        else:
            return self.pagify(X, 1, N, T, I)

    def _matrix_to_tensor(self, X, N, T, I):
        if self.L:
            return self.unpagify(X, self.L, N, T, I)
        else:
            return self.unpagify(X, 1, N, T, I)

    @staticmethod
    def pagify(tensor, L, N, T, I):
        m = T // L
        assert T % L == 0, "choose L that is multiple of T"
        pagified_tensor = np.zeros([L * N, m * I])
        for n in range(N):
            series = tensor[n, :, :]
            pagified = series.reshape([L, m * I], order="F")
            pagified_tensor[L * n : L * (n + 1), :] = pagified
        return pagified_tensor

    @staticmethod
    def unpagify(matrix, L, N, T, I):
        tensor = np.zeros([N, T, I])
        for n in range(N):
            series = matrix[L * n : L * (n + 1), :].reshape([T, I], order="F")
            tensor[n, :, :] = series
        return tensor

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

        max_t = max(list(time_dict.values())) + 1
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

    def _compress_tensor(self, tensor):
        # Apply Alternating Least Squares to decompose the tensor into CP form
        als_model = ALS(k_factors=self.k_factors)

        # Modifies tensor by filling nans with zeros
        als_model.fit(tensor)

        # Only save the ALS output tensors
        return als_model.cp_factors

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

    @staticmethod
    def get_tensor_from_factors(
        factors: ndarray,
        unit_idx: Optional[List[int]] = None,
        time_idx: Optional[List[int]] = None,
    ):
        tensor = ALS._predict(factors, unit_idx=unit_idx, time_idx=time_idx)
        return tensor

    def _compute_drift(self, new_tensor, Y_new, regime):
        # transfrom tensor to matrix (NI X T)
        N, T, I = new_tensor.shape
        matrix = new_tensor.transpose([2, 0, 1]).reshape([N * I, T])

        # get factors and predict
        X = regime.tensor_cp_combined_action_unit_factors
        predictions = X @ Y_new.T

        # compute error
        true = matrix.copy()
        predicion_error = np.nanmean(
            np.square((predictions - true)), axis=0
        ) / np.nanmean(np.square(true), axis=0)

        assert np.isnan(predicion_error).sum() == 0
        return predicion_error

    def partial_fit(
        self,
        new_df: pd.DataFrame,
    ) -> None:
        time_pointer = self.T
        current_regime = self.regimes[-1]

        if self.verbose:
            print(
                f"current regime: {current_regime.index}, started at: {current_regime.start_time}"
            )

        # check whether we have used enough data points to train the model
        ## None means it has reached full update state
        if self.current_regime_tensor is None:
            partial_update = True
        ## else check condition
        else:
            no_training_points = self.current_regime_tensor.shape[1]
            partial_update = no_training_points >= self.full_training_time_steps

        # load tensor
        new_tensor = self._get_partial_tensor(new_df)

        if partial_update:
            if self.verbose:
                print("partial_update")
            # if so just update factors
            Y_new = self._compute_updated_factors(new_tensor, current_regime)
            distance_error = self._compute_drift(new_tensor, Y_new, current_regime)
            self.distance_error = np.concatenate([self.distance_error, distance_error])
            self._update_cusum(current_regime)
            regime_shift, shift_time = self._check_regime_shift(current_regime)
            if regime_shift:
                if self.verbose:
                    print(f"regime shift, at {shift_time}")
                # update factors until shift_time
                non_shift_period = shift_time - time_pointer
                if non_shift_period:
                    self._update_time_factors(
                        Y_new[:non_shift_period, :], current_regime.index
                    )
                # init new regime and append
                self.regimes[-1].end_time = shift_time - 1
                new_regime = Regime(len(self.regimes), shift_time)
                self.regimes.append(new_regime)

                # fit using new regime
                tensor = new_tensor[:, non_shift_period:, :]
                self._fit(tensor, new_regime.index)

            else:
                self._update_time_factors(Y_new, current_regime.index)

            # update nan mask
            self._update_nan_mask(new_tensor)

        else:
            # else fully train again
            tensor = np.concatenate([self.current_regime_tensor, new_tensor], axis=1)
            self._fit(tensor, current_regime.index)
            self.current_regime_tensor = tensor
            # delete current regime if full training status is reached
            if self.current_regime_tensor.shape[1] > self.full_training_time_steps:
                self.current_regime_tensor = None

    def _update_cusum(self, regime):
        current_step = len(self.distance_error)
        cusum_index = len(self.cusum)
        self.cusum = np.concatenate([self.cusum, np.zeros(current_step - cusum_index)])

        for t in range(cusum_index, current_step):
            self.cusum[t] = max(
                self.cusum[t - 1] + self.distance_error[t] - regime.mean_drift,
                0,
            )

    def _check_regime_shift(self, regime):
        threshold = self.threshold_multiplier * regime.mean_drift
        check = self.cusum[regime.start_time :] > threshold
        shift = check.sum() > 0
        shift_time = None
        if shift:
            shift_time = np.argmax(check) + regime.start_time
        return shift, shift_time

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

    def _compute_updated_factors(self, new_tensor: ndarray, regime: Regime):
        """update time factors using the new observations in new tensor ([N x new_timesteps x I])

        Args:
            new_tensor (ndarray): new tensor that correspond to new obseravations

        Returns:
            ndarray: new time factors (new_timesteps x k)
        """
        # check if self.tensor_cp_factors is not None
        assert (
            regime.tensor_cp_factors is not None
        ), "regime.tensor_cp_factors is None, have you called fit()?"

        # transfrom tensor to matrix (NI X T)
        N, T, I = new_tensor.shape
        matrix = new_tensor.transpose([2, 0, 1]).reshape([N * I, T])

        # update factors based on new columns
        Y = regime.tensor_cp_factors[1]
        X = regime.tensor_cp_combined_action_unit_factors
        Y_new = self.get_new_time_factors(matrix, Y, X, self.min_singular_value)
        return Y_new

    def _update_time_factors(self, Y_new, regime_index):
        regime = self.regimes[regime_index]
        regime.tensor_cp_factors[1] = np.concatenate(
            [regime.tensor_cp_factors[1], Y_new], axis=0
        )
        assert (
            regime.tensor_cp_factors[1].shape[1] == regime.tensor_cp_factors[0].shape[1]
        )

    @staticmethod
    def get_new_time_factors(matrix, Y, X, min_singular_value):
        k = Y.shape[1]
        Y_new = np.zeros([matrix.shape[1], k])
        for col_i in range(matrix.shape[1]):
            col = matrix[:, col_i]
            observed_entries = np.argwhere(~np.isnan(col)).flatten().astype(int)
            mat = X[observed_entries, :].T @ X[observed_entries, :]
            b = X[observed_entries, :].T @ col[observed_entries]
            assert b.shape == (k,), b.shape
            assert mat.shape == (k, k), mat.shape
            Y_new[col_i, :] = np.linalg.lstsq(mat, b, rcond=min_singular_value)[0]
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
            "regimes": model_tuple.regimes,
            "true_intervention_assignment_matrix": model_tuple.true_intervention_assignment_matrix,
            "unit_column": model_tuple.unit_column,
            "time_column": model_tuple.time_column,
            "actions": model_tuple.actions,
            "metric": model_tuple.metric,
            "cusum": model_tuple.cusum,
            "distance_error": model_tuple.distance_error,
            "current_regime_tensor": model_tuple.current_regime_tensor,
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
        regimes = loaded_dict.pop("regimes").tolist()
        true_intervention_assignment_matrix = loaded_dict.pop(
            "true_intervention_assignment_matrix"
        )
        unit_column = loaded_dict.pop("unit_column")
        time_column = loaded_dict.pop("time_column")
        actions = loaded_dict.pop("actions").tolist()
        metric = loaded_dict.pop("metric")
        cusum = loaded_dict.pop("cusum")
        distance_error = loaded_dict.pop("distance_error")
        current_regime_tensor = loaded_dict.pop("current_regime_tensor")

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
        self.regimes = regimes
        self.true_intervention_assignment_matrix = true_intervention_assignment_matrix
        self.unit_column = unit_column
        self.time_column = time_column
        self.actions = actions
        self.metric = metric
        self.tensor_nans = tensor_nans
        self.cusum = cusum
        self.distance_error = distance_error
        self.current_regime_tensor = current_regime_tensor
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
