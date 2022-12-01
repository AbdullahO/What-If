"""
Generate symnthetic data according to the tensor factor model in  https://arxiv.org/abs/2006.07691
"""

import numpy as np
import pandas as pd
import tensorly as tl
from sklearn.cluster import KMeans
import os
import json
from typing import List, Optional, Any
from numpy import ndarray

# set backend to numpy explicitly
tl.set_backend("numpy")

ROUND_FLOAT = 7  # decimal places


class UnitCov:
    """class for defining a unit covariate"""

    def __init__(
        self,
        name: str,
        discrete: Optional[bool] = True,
        categories: Optional[ndarray] = None,
        cov_range: Optional[list] = None,
    ):
        """_summary_

        Args:
            name (str): name of the covariate (e.g., gender)
            discrete (bool, optional): True if discrete. Defaults to True.
            categories (list, optional): categories of cov if discrete. If None is given, it will be generated randomly.
                                         Defaults to None.
            cov_range (list, optional): covariate range if continous. Defaults to None.
        """
        self.name = name
        self.discrete = discrete
        self.categories = None
        if categories is None:
            n_c = np.random.choice(np.arange(2, 6), p=[0.3, 0.3, 0.2, 0.2])
            categories = np.arange(n_c)
        if self.discrete:
            self.categories = categories
        # continous notImplemented
        else:
            if cov_range is None:
                cov_range = [0, 1]
            self.range = cov_range
        self.unit_labels = None


class Metric:
    def __init__(
        self,
        name: str,
        metric_range: Optional[list] = None,
        difference_metric: Optional[bool] = False,
        init_values_range: Optional[List[int]] = None,
        clip_range: Optional[list] = None,
    ):
        """base class for metric

        Args:
            name (str): metric name
            metric_range (list, optional): min and max of the metric range. If None, the metric range is set to [0,1]. Defaults to None.
            difference_metric (bool, optional): if True, the metric difference (e.g., change in sales) will be modeled as a low rank tesnor instead of the metric itself. Defaults to False.
            init_values_range (bool, optional): if differennce is True, this rannge will be used to choose initial value for metric. Defaults to None.
            clip_range (list, optional): Hard threshold on the metric values. The metric range may be changes once effects are introduced. the clip range ensures that the value is clipped to be in this range. Defaults to None.
        """
        self.name = name
        if metric_range is None:
            metric_range = [0, 1]
        self.range = metric_range
        self.difference_metric = difference_metric
        if init_values_range is None:
            init_values_range = [0, 1]
        self.init_values_range = init_values_range
        self.init_values = None
        self.clip_range = clip_range


class IntCov:
    def __init__(
        self,
        name: str,
        discrete: Optional[bool] = None,
        categories: Optional[list] = None,
        divisions: Optional[int] = None,
        cov_range: Optional[list] = None,
        assignment: Optional[list] = None,
    ):
        """base class for intervention covariates

        Args:
            name (str): intervention name
            discrete (bool, optional): True if discrete. Defaults to None.
            categories (list, optional): categories of cov if discrete. If None is given, it will be generated randomly.
                                         Defaults to None.
            cov_range (list, optional): covariate range if continous. Defaults to None.
            divisions (int, optional): Number of partitions for this covaraite that will be assigned to a particular intervention.
                                        For example, if categories = [1,2,3], setting divisions = 2 means that two of the categories (e.g., [1,2]) will correspond to the same intervention and the third category will correspond to another intervention
                                        Defaults to None.
            assignment (list, optional): If discrete, assign categories to divisions. Defaults to None.
        """
        self.name = name
        if categories is not None:
            discrete = True
        if discrete is None:
            discrete = np.random.random() < 0.5
        self.discrete = discrete
        self.categories = None
        self.divisions_threshold = None
        self.division_categories = None
        self.range = None
        self.assignment = assignment
        if self.discrete:
            self._get_discrete_attributes(divisions, categories)

        else:
            self._get_cont_attributes(divisions, cov_range)

    def _get_cont_attributes(self, divisions, cov_range):
        if not divisions:
            self.divisions = np.random.randint(1, 4)
        else:
            self.divisions = divisions

        if cov_range is not None:
            self.range = cov_range
        else:
            self.range = sorted([np.random.randn() * 100, np.random.randn() * 100])

        self._get_division_thresholds()

    def _get_discrete_attributes(self, divisions, categories):
        # generate divisions if none are given
        if not divisions:
            # if categories are given, divisions must be < len(categories)
            if categories is not None:
                self.divisions = np.random.randint(1, len(categories))
            # else choose [1,4] (arbitrary number)
            else:
                self.divisions = np.random.randint(1, 4)
        else:
            self.divisions = divisions

        if categories is not None:
            self.categories = categories
        else:
            self.categories = np.arange(self.divisions)
        # check that assignment is valid
        if self.assignment:
            assert (
                max(self.assignment) == divisions - 1
            ), f"maximum integer in assignment should equal to the number of divisions - 1, {max(self.assignment)} > {divisions-1}"

        # get division categories (i.e., which division correspond to which categories)
        self._get_division_categories()

    def _get_division_thresholds(self):
        self.divisions_threshold = sorted(
            np.linspace(self.range[0], self.range[1], self.divisions + 1)
        )
        self.divisions_labels = [
            f"{self.divisions_threshold[d]} <= {self.name} < {self.divisions_threshold[d+1]}"
            for d in range(self.divisions)
        ]

    def _get_division_categories(self):
        if self.assignment:
            self.assignment = np.array(self.assignment)
            self.division_categories = [
                [self.categories[j] for j in list(np.where(self.assignment == i)[0])]
                for i in range(self.divisions)
            ]
        else:
            self.division_categories = _split_categories(
                self.categories, self.divisions
            )

        # check
        assert len(self.division_categories) == self.divisions
        self.divisions_labels = [
            f"{self.name} = {cat_list}" for cat_list in self.division_categories
        ]


def _split_categories(categories: list, divisions: int):
    """split categories into divisions at random

    Args:
        categories (_type_): _description_
        divisions (_type_): _description_
        indices (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """

    subgroups = []
    no_categoires = len(categories)
    assert no_categoires >= divisions
    assert divisions > 0
    cat_shuf = list(categories)
    np.random.shuffle(cat_shuf)
    indices = sorted(
        np.random.choice(
            np.arange(no_categoires - 1), replace=False, size=divisions - 1
        )
        + 1
    )
    start = 0
    for ind in list(indices) + [no_categoires]:
        subgroups.append(cat_shuf[start:ind])
        start = ind
    return subgroups


class SyntheticDataModule:
    """
    SyntheticDataModule class
    """

    def __init__(
        self,
        num_units: int,
        max_timesteps: int,
        num_interventions: int,
        metrics: List[Metric],
        unit_cov: Optional[list] = None,
        int_cov: Optional[list] = None,
        rank: Optional[int] = 2,
        freq: Optional[str] = None,
        start: Optional[pd.Timestamp] = pd.Timestamp("01/01/2020 00:00"),
    ) -> None:
        """_summary_

        Args:
            num_units (int): number of units in your data
            max_timesteps (int): number of maximum timesteps you will generate in your data
            num_interventions (int): number of interventions
            metrics (List[Metric]): list of metrics
            unit_cov (List[UnitCov], optional): list of unit covaraites. Defaults to None.
            int_cov (List[IntCov], optional): list of interventions covaraites. Defaults to None.
            rank (int, optional): rank of the produced tensor. Defaults to 2.
            freq (str, optional): frequency of the time steps. If none, integers would be used. Defaults to None.
        """
        self.num_units = num_units
        self.max_timesteps = max_timesteps
        self.num_interventions = num_interventions
        self.num_metrics = len(metrics)
        self.metrics = metrics
        self.effects = []
        self.freq = freq
        self.factors = None
        self.U, self.T, self.I, self.M = (None, None, None, None)
        self.metric_columns = None
        self.intervention_columns = None
        self.ss_tensor = None
        self.ss_df = None
        self.unit_cov = unit_cov
        self.int_cov = int_cov
        self.rank = rank
        self.metrics_range = [metric.range for metric in metrics]
        self.assignments_labels = None
        self.assignments = None
        self.subpopulations = []
        self.subpopulations_funcs = []
        self.tensor_min = 0
        self.tensor_max = 0
        self.trend_coeff = None
        self.time_factor_periods = None
        self.time_factor_har_amps = None
        self.time_factor_har_shifts = None
        self.poly_coeff = None
        self.start = start

    def generate_init_factors(
        self,
        max_periods=50,
        max_amp_har=1,
        min_amp_har=-1,
        periodic=True,
        lin_tren=False,
        trend_coeff=None,
        poly_trend=False,
        poly_coeff=None,
    ):
        self._generate_factors(
            max_periods,
            max_amp_har,
            min_amp_har,
            periodic,
            lin_tren,
            trend_coeff,
            poly_trend,
            poly_coeff,
        )
        self.factors = self.U, self.T, self.I, self.M

        if self.int_cov:
            self._init_intervention_covs()
        if self.unit_cov:
            self._init_unit_covs()

    def add_effects(self, effects):
        """_summary_

        Args:
            effects (list): _description_
        """
        self.effects += effects

    def generate(self, time_range):
        t0, t1 = time_range
        # construct tensor
        tensor = tl.cp_to_tensor(
            (np.ones(self.rank), [self.U, self.T[t0 : t1 + 1, :], self.I, self.M])
        )

        # Adjust metric ranges
        if self.metrics_range is not None:
            for m in range(self.num_metrics):
                min_, max_ = self.metrics_range[m]
                tensor[:, :, :, m] -= self.tensor_min
                tensor[:, :, :, m] /= self.tensor_max
                tensor[:, :, :, m] *= max_ - min_
                tensor[:, :, :, m] += min_

        df = self._init_df(t0, t1)
        if self.int_cov:
            df = self._add_unit_cov(df)
        if self.unit_cov:
            df = self._generate_int_cov(df)
        tensor = self.genereate_effects(tensor)
        df = self._add_metrics(df, tensor)
        return tensor, df

    def genereate_effects(self, tensor):
        for effect in self.effects:
            metric = effect["metric"]
            intervention_number = effect["intervention"]
            subpopulation = effect["subpop"]
            effect_size = effect["effect"]
            if subpopulation is None:
                subpopulation = np.ones(self.U.shape[0]).astype(bool)
            else:
                subpopulation = subpopulation()
            metric_index = self.metrics.index(metric)
            control = tensor[subpopulation, :, 0, metric_index].mean(0)
            c_eff_size = tensor[
                subpopulation, :, intervention_number, metric_index
            ].mean(0)
            tensor[
                subpopulation, :, intervention_number, metric_index
            ] += effect_size * control - (c_eff_size - control)
            noise = (
                0.01
                * tensor.mean()
                * np.random.normal(
                    size=tensor[
                        subpopulation, :, intervention_number, metric_index
                    ].shape
                )
            )
            tensor[subpopulation, :, intervention_number, metric_index] += noise
        return tensor

    def auto_subsample(self, periods, tensor, df):

        """
        return intervention assignments (num_units x num_timesteps) and mask for
        observations (num_units x num_timesteps) then use sample to generate df_ss and tesnor_ss
        """
        T = tensor.shape[1]
        int_idx = np.zeros([self.num_units, T]).astype(int)
        obs_idx = np.zeros([self.num_units, T]).astype(bool)
        start = 0

        for period in periods:
            until = period["until"]
            assert (
                until > start
            ), f"value for until should be greater than that of previous periods; {until} is less than {start}"
            assert (
                until <= T
            ), f"value for until should be less than or equal to total number of time steps; {until} is greater than {T}"

            assignment = period["intervention_assignment"]
            if "assignment_subpop" in period:
                assignment_subpop = period["assignment_subpop"]
            else:
                assignment_subpop = None
            # assign interventions
            int_idx[:, start:until] = self._auto_assign(
                assignment, assignment_subpop, until - start
            )

            if "observations_selection" in period:
                observation_selection = period["observations_selection"]
            else:
                observation_selection = "random"
            if "fraction_of_observed_values" in period:
                fraction_of_observed_values = period["fraction_of_observed_values"]
            else:
                fraction_of_observed_values = 1.0

            obs_idx[:, start:until] = self._auto_mask(
                observation_selection, fraction_of_observed_values, until - start
            )  # subsample observations
            start = until

        ss_tensor, ss_df = self.sample(int_idx, obs_idx, tensor, df)
        return ss_tensor, ss_df

    def sample(self, intervention_assignments, observation_mask, tensor, df):

        intervention_assignments = intervention_assignments.flatten()
        T = tensor.shape[1]
        # subsample tensor intervention
        mesh = np.array(np.meshgrid(np.arange(self.num_units), np.arange(T)))
        combinations = mesh.T.reshape(-1, 2)
        self.mask = np.full(tensor.shape, 0)
        self.mask[
            combinations[:, 0], combinations[:, 1], intervention_assignments, :
        ] = 1
        subsampled_tensor = tensor[
            combinations[:, 0], combinations[:, 1], intervention_assignments, :
        ].reshape(self.num_units, T, self.num_metrics)
        subsampled_tensor[observation_mask] = np.nan

        # subsample df
        df_copy = df.copy()
        df_copy["intervention"] = intervention_assignments

        # select metric column according to intervention_assignments
        for i, metric_list in enumerate(self.metric_columns):
            metric_matrix = df_copy.sort_values(["unit_id", "time"])[metric_list].values
            chosen_metric = metric_matrix[
                np.arange(df.shape[0]), intervention_assignments
            ]
            if self.metrics[i].difference_metric:
                chosen_metric = chosen_metric.reshape([self.num_units, T])
                chosen_metric = (
                    chosen_metric.cumsum(axis=1) + self.metrics[i].init_values
                )
                chosen_metric = chosen_metric.flatten()

            if self.metrics[i].clip_range is not None:
                min_, max_ = self.metrics[i].clip_range
                chosen_metric = np.clip(chosen_metric, min_, max_)
            chosen_metric[observation_mask.flatten()] = np.nan
            df_copy[self.metrics[i].name] = chosen_metric

            df_copy = df_copy.drop(metric_list, axis="columns")

        # select intervention covariates according to intervention_assignments
        if self.int_cov:
            for i, intervention_list in enumerate(self.intervention_columns):
                int_matrix = df_copy[intervention_list].values
                chosen_cov = int_matrix[
                    np.arange(df.shape[0]), intervention_assignments
                ]
                df_copy[self.int_cov[i].name] = chosen_cov
                df_copy = df_copy.drop(intervention_list, axis="columns")

        df_copy = df_copy.dropna()
        return subsampled_tensor, df_copy

    def _auto_mask(self, observation_selection, fraction_of_observed_values, timesteps):
        if fraction_of_observed_values < 1.0:
            if observation_selection == "random":
                idx_obs = (
                    np.random.random(size=(self.num_units, timesteps))
                    >= fraction_of_observed_values
                )
            elif observation_selection == "tail":
                idx_obs = np.random.random(size=(self.num_units, timesteps))
                chosen_units = (
                    np.random.random(size=(self.num_units,))
                    >= fraction_of_observed_values
                )
                time_tail = np.random.randint(
                    timesteps // 4, timesteps, size=chosen_units.sum()
                )
                idx_obs[chosen_units, time_tail] = np.nan
                idx_obs = np.isnan(np.cumsum(idx_obs, axis=1))
            else:
                raise Exception("selection must be either random, or tail")
        else:
            idx_obs = np.zeros([self.num_units, timesteps]).astype(bool)
        return idx_obs

    def _auto_assign(self, selection, selection_subpop, timesteps):
        if selection == "random":
            idx = np.random.choice(
                np.arange(self.num_interventions), size=(self.num_units, timesteps)
            )
            idx = idx.astype(int)
        elif selection == "control":
            idx = np.zeros((self.num_units, timesteps))
            idx = idx.astype(int)

        elif selection == "random_unit":
            idx_units = np.random.choice(
                np.arange(self.num_interventions), size=(self.num_units, 1)
            )
            idx = np.zeros([self.num_units, timesteps])
            idx[:, :] = idx_units
            idx = idx.astype(int)
        elif selection == "cov_unit":
            idx_units = np.random.choice(
                np.arange(self.num_interventions), size=(self.num_units, 1)
            )
            assert selection_subpop is not None, "cov_unit requires selection_subpop"
            subpopulations = np.zeros(
                [self.num_units, len(selection_subpop) + 1]
            ).astype(bool)
            p_i = 0
            for subpop, p in selection_subpop.items():
                assert len(p) == self.num_interventions
                subpopulations[:, p_i] = subpop()
                assert (
                    subpopulations.sum(1).max() <= 1
                ), "subpops are not mutually execlusive!"
                subpop_size = int(subpopulations[:, p_i].sum())
                idx_units[subpopulations[:, p_i], 0] = np.random.choice(
                    np.arange(self.num_interventions), size=subpop_size, p=p
                )
                p_i += 1
            subpopulations[:, -1] = subpopulations.sum(1) == 0
            subpop_size = subpopulations[:, -1].sum()
            idx_units[subpopulations[:, -1], 0] = np.random.choice(
                np.arange(self.num_interventions), size=subpop_size
            )
            idx = np.zeros([self.num_units, timesteps])
            idx[:, :] = idx_units
            idx = idx.astype(int)
        else:
            raise Exception(
                "selection must be either random, random_unit, cov_unit, or control"
            )
        return idx

    def _init_df(self, t0, t1):
        """initialise df with unit and time columns"""
        df = pd.DataFrame(columns=["unit_id", "time"])
        if self.freq is not None:
            td = pd.to_timedelta(self.freq)
            time = pd.date_range(
                start=self.start + td * t0,
                freq=self.freq,
                periods=t1 - t0 + 1,
            )
        else:
            time = np.arange(t0, t1 + 1)

        # populate unit and time
        mesh = np.array(np.meshgrid(np.arange(self.num_units), time))
        combinations = mesh.T.reshape(-1, 2)
        df["unit_id"] = combinations[:, 0]
        df["time"] = combinations[:, 1]
        if self.freq is not None:
            df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values(["unit_id", "time"])

        return df

    def _init_intervention_covs(self):
        no_int_cov = len(self.int_cov)
        self.intervention_columns = [list() for _ in range(no_int_cov)]
        no_interventions, _ = self.I.shape

        ## assign interventions to divisions
        ## e.g., if int_cov is discrete and \in {0,1}, the corresponding column in the assignment
        # matrix tells us which intervention maps to which cov value.
        assignment = np.zeros([no_interventions, no_int_cov]).astype(int)
        while np.unique(assignment, axis=0).shape[0] < no_interventions:
            no_divisions = [cov.divisions for cov in self.int_cov]
            for cov in range(no_int_cov):
                if self.int_cov[cov].assignment is None:
                    assignment[:, cov] = np.random.choice(
                        np.arange(no_divisions[cov]), size=no_interventions
                    )
                else:
                    assert (
                        len(self.int_cov[cov].assignment) == no_interventions
                    ), f"assignment for {self.int_cov[cov].name}'s length must equal the no. interventions"
                    assignment[:, cov] = self.int_cov[cov].assignment
        self.assignments = assignment
        self.assignments_labels = np.zeros_like(assignment).astype("object")

        # add actions labels
        self.actions = []
        for i in range(no_interventions):
            action = {"name": f"action_{i}", "id": i, "conditions": []}
            for c_i, cov in enumerate(self.int_cov):
                if len(np.unique(self.assignments[:, c_i])) == 1:
                    continue
                cond = {"column": cov.name}
                cont = not cov.discrete
                div = self.assignments[i, c_i]
                if cont:
                    cond["type"] = "range"
                    cond["lower"] = cov.divisions_threshold[div]
                    cond["upper"] = cov.divisions_threshold[div + 1]
                else:
                    cond["type"] = "categorical"
                    cond["values"] = cov.division_categories[div]
                action["conditions"].append(cond)
            self.actions.append(action)

    def _generate_int_cov(self, df):
        # generate interventions covariates
        no_interventions = self.I.shape[0]
        cov_names = [cov.name for cov in self.int_cov]

        for c_i, cov in enumerate(self.int_cov):
            divisions = cov.divisions_threshold
            cont = not cov.discrete
            for i in range(no_interventions):
                div = self.assignments[i, c_i]
                if cont:
                    intervention_cov_range = [divisions[div], divisions[div + 1]]
                    df[f"{cov_names[c_i]}_{i}"] = np.random.uniform(
                        low=intervention_cov_range[0],
                        high=intervention_cov_range[1],
                        size=df.shape[0],
                    )
                else:
                    df[f"{cov_names[c_i]}_{i}"] = np.random.choice(
                        cov.division_categories[div], size=df.shape[0]
                    )
                self.intervention_columns[c_i].append(f"{cov_names[c_i]}_{i}")
            dict_ = dict(zip(np.arange(cov.divisions), cov.divisions_labels))
            self.assignments_labels[:, c_i] = np.vectorize(dict_.__getitem__)(
                self.assignments[:, c_i]
            )
        return df

    def _init_unit_covs(self):
        _, rank = self.U.shape

        for cov in self.unit_cov:
            i = np.random.randint(2) + 1
            rs = np.random.choice(np.arange(rank), size=i, replace=False)
            X = self.U[:, rs]
            if cov.discrete:
                n_c = len(cov.categories)
                kmeans = KMeans(n_clusters=n_c, random_state=0).fit(X)
                labels = dict(zip(np.arange(n_c), cov.categories))
                categories_label = np.vectorize(labels.__getitem__)(kmeans.labels_)
                cov.unit_labels = categories_label
            else:
                beta = np.random.random([i, 1])
                labels = X @ beta
                labels -= labels.min()
                labels /= labels.max()
                labels *= cov.range[1] - cov.range[0]
                labels += cov.range[0]
                cov.unit_labels = labels[:, 0]

    def _add_unit_cov(self, df):
        no_units, _ = self.U.shape
        cov_names = [cov.name for cov in self.unit_cov]
        for c_i, cov in enumerate(self.unit_cov):
            dict_labels = dict(zip(np.arange(no_units), cov.unit_labels))
            df[f"{cov_names[c_i]}"] = df["unit_id"].map(dict_labels)
        return df

    def _add_metrics(self, df, tensor):
        self.metric_columns = [list() for _ in range(self.num_metrics)]
        # populate metrics
        for m, metric in enumerate(self.metrics):
            change_str = ""
            if metric.difference_metric:
                change_str = "_change"
            for i in range(self.num_interventions):
                df[f"{metric.name}{change_str}_{str(i)}"] = tensor[:, :, i, m].flatten()
                self.metric_columns[m].append(f"{metric.name}{change_str}_{str(i)}")
            if metric.difference_metric and metric.init_values is None:
                min_, max_ = metric.init_values_range
                metric.init_values = (
                    np.random.random(size=(self.num_units, 1)) * (max_ - min_) + min_
                )
        return df

    def _generate_factors(
        self,
        max_periods=50,
        max_amp_har=1,
        min_amp_har=0,
        periodic=True,
        lin_tren=False,
        trend_coeff=None,
        poly_trend=False,
        poly_coeff=None,
    ):
        # Generate factors
        self.U = np.random.random([self.num_units, self.rank])
        self.T = np.random.random([self.max_timesteps, self.rank]) - 0.5
        self.I = np.random.random([self.num_interventions, self.rank])
        self.M = np.random.random([self.num_metrics, self.rank])

        # generate temporal factors
        if periodic:
            self.time_factor_periods = 1 / (np.random.random(self.rank) * max_periods)
            self.time_factor_har_amps = (
                np.random.random(self.rank) * (max_amp_har - min_amp_har) + min_amp_har
            )
            self.time_factor_har_shifts = np.random.random(self.rank) * np.pi
            for r in range(self.rank):
                self.tensor_max += self.time_factor_har_amps[r] + 0.5
                self.T[:, r] += self.time_factor_har_amps[r] * np.sin(
                    self.time_factor_har_shifts[r]
                    + 2
                    * np.pi
                    * self.time_factor_periods[r]
                    * np.arange(self.max_timesteps)
                    / (self.max_timesteps)
                )
        if lin_tren:
            if trend_coeff is None:
                trend_coeff = (
                    0.05 * (np.random.random(self.rank) - 0.5) / self.max_timesteps
                )
            self.trend_coeff = trend_coeff
            for r in range(self.rank):
                self.T[:, r] += self.trend_coeff * np.arange(self.max_timesteps)
        if poly_trend:
            if poly_coeff is None:
                poly_coeff = (
                    5 * (np.random.random(self.rank) - 0.5) / (self.max_timesteps**2)
                )
            self.poly_coeff = poly_coeff
            self.T[:, r] += self.trend_coeff * np.arange(self.max_timesteps) ** 2
        self.tensor_max = np.abs(self.T).max()
        self.tensor_max = tl.cp_to_tensor(
            (
                np.ones(self.rank),
                [self.U, self.tensor_max * np.ones([1, self.rank]), self.I, self.M],
            )
        ).max()
        self.tensor_min = -1 * self.tensor_max

    def export(self, name, tensor, ss_df, dir=""):
        # create dir
        path = os.path.join(dir, name)
        if not os.path.exists(path):
            os.makedirs(path)

        # save subsampled df
        ss_df.drop("intervention", axis="columns").round(ROUND_FLOAT).to_csv(
            f"{path}/{name}.csv", index=False
        )

        # save tensor as npy
        np.save(f"{path}/tensor.npy", tensor.round(ROUND_FLOAT))
        np.save(f"{path}/mask.npy", self.mask)
