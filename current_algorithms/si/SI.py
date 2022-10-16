import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from src.cvxRegression import ConvexRegression
import statsmodels.api as sm
import seaborn as sns
from src.diagnostic import diagnostic_test
from src.algorithms import hsvt_ols
from src.mSSA import mSSA
from src.matrix import approximate_rank
import matplotlib.pyplot as plt
from scipy.stats import norm


class SI(object):
    """docstring for SI"""

    def __init__(
        self,
        center=True,
        rcond=1e-15,
        t=0.99,
        alpha=0.05,
        include_pre=True,
        rank_method="spectral_energy",
        return_donors_info=True,
        mSSA_CI=False,
        use_lasso=False,
        use_ridge=False,
    ):
        super(SI, self).__init__()
        self.center = center
        self.rcond = rcond
        self.t = t
        self.alpha = alpha
        self.include_pre = include_pre
        self.rank_method = rank_method
        self.return_donors_info = return_donors_info
        self.pre_df = None
        self.post_df = None
        self.synthetic_results = None
        self.donors_info = None
        self.diagnosis = None
        self.non_donor_list = []
        self.std_df_mSSA = None
        self.std_df = None
        self.use_lasso = use_lasso
        self.use_ridge = use_ridge

    def diagnose(self, pre_df, post_df, interventions=None):
        columns = ["unit", "intervention", "metric"]

        # sort dataframes
        pre_df = pre_df.sort_values(by=columns)
        post_df = post_df.sort_values(by=columns)

        # get ivs, units, metrics
        if interventions is None:
            ivs = np.sort(pd.unique(post_df.intervention))
        else:
            ivs = list(
                set(np.sort(pd.unique(post_df.intervention))).intersection(
                    interventions
                )
            )
            # print(ivs)
        units = list(np.sort(pd.unique(pre_df.unit)))
        metrics = list(np.sort(pd.unique(pre_df.metric)))

        # get number of units and interventions
        N, K, M = len(units), len(ivs), len(metrics)

        # initialize
        diagnostic_rslts = np.empty((K * M * N, 3))
        diagnostic_rslts[:] = np.nan

        # perform diagnostic tests
        for u, unit in enumerate(units):
            for i, iv in enumerate(ivs):
                unit_ids = pd.unique(post_df[post_df.intervention == iv]["unit"])
                unit_ids = unit_ids[unit_ids != unit]
                for m, metric in enumerate(metrics):
                    if len(unit_ids) <= 1:
                        diagnostic_rslts[u * (M * K) + i * M + m, :] = [
                            np.nan,
                            np.nan,
                            np.nan,
                        ]
                    else:
                        diagnostic_rslts[u * (M * K) + i * M + m, :] = diagnostic_test(
                            pre_df,
                            post_df,
                            unit_ids,
                            metric,
                            iv,
                            t=self.t,
                            alpha=self.alpha,
                        )

        # create output dataframe
        df = pd.DataFrame(
            data=diagnostic_rslts,
            columns=["pvalues_test", "energy_statistic", "energy_statistic_value"],
        )
        diag_ivs = [ivs[k // M] for k in range(K * M)] * N
        diag_metrics = metrics * K * N
        diag_units = [units[k // (M * K)] for k in range(N * K * M)]
        df.insert(0, "metric", diag_metrics)
        df.insert(0, "intervention", diag_ivs)
        df.insert(0, "unit", diag_units)
        df["pvalues_test"] = df["pvalues_test"].replace(0, "Fail")
        df["pvalues_test"] = df["pvalues_test"].replace(1, "Pass")
        df["pvalues_test"] = df["pvalues_test"].replace(np.nan, "N/A")
        df["energy_statistic"] = df["energy_statistic"].replace(0, "Fail")
        df["energy_statistic"] = df["energy_statistic"].replace(1, "Pass")
        df["energy_statistic"] = df["energy_statistic"].replace(np.nan, "N/A")
        self.diagnosis = df

    def fit(
        self, pre_df, post_df, non_donor_list=[], mSSA_CI=False, interventions=None
    ):
        columns = ["unit", "intervention", "metric", "donor"]
        # sort dataframes by (unit, intervention)
        self.pre_df = pre_df.sort_values(by=columns[:3])
        self.post_df = post_df.sort_values(by=columns[:3])

        # Add donors column, and filter non-donors
        if "donor" not in self.pre_df.columns:
            self.pre_df.insert(0, "donor", 1)
        self.pre_df.loc[pre_df.unit.isin(non_donor_list), "donor"] = 0
        if "donor" not in self.post_df.columns:
            self.post_df.insert(0, "donor", 1)
        self.post_df.loc[pre_df.unit.isin(non_donor_list), "donor"] = 0
        self.non_donor_list = non_donor_list

        # get all unique interventions (from post-intervention dataframe)
        if interventions is None:
            ivs = np.sort(pd.unique(self.post_df.intervention))
        else:
            ivs = set(np.sort(pd.unique(self.post_df.intervention))).intersection(
                interventions
            )

        # get all units (using pre-intervention data)
        units = list(np.sort(pd.unique(self.pre_df.unit)))

        # get all metrics
        metrics = list(np.sort(pd.unique(self.pre_df.metric)))

        # get number of units and interventions
        N, K, M = len(units), len(ivs), len(metrics)
        T0 = self.pre_df.shape[1] - len(columns)
        T = T0 + self.post_df.shape[1] - len(columns)

        # check to make sure there aren't any duplicate units in the pre-intervention dataframe
        assert len(self.pre_df.unit.unique()) == N

        # initialize output dataframe
        yh_data = np.array([])
        idx_data = np.array([])
        std_data = np.array([])
        donors_dict = {}
        for iv in ivs:
            donors_dict[iv] = {}
            # get potential donors (units who receive intervention 'iv') from POST-intervention data
            unit_ids = pd.unique(
                self.post_df[
                    (self.post_df.intervention == iv) & (self.post_df.donor == 1)
                ]["unit"]
            )

            for unit in units:
                # filter donors who received a differnt pre intervention
                pre_iv = self.pre_df[self.pre_df.unit == unit]["intervention"].values[0]
                units_same_pre_int = pd.unique(
                    self.pre_df[(self.pre_df.intervention == pre_iv)]["unit"]
                )
                unit_ids = list(
                    set(
                        pd.unique(
                            self.post_df[
                                (self.post_df.intervention == iv)
                                & (self.post_df.donor == 1)
                            ]["unit"]
                        )
                    ).intersection(units_same_pre_int)
                )
                unit_ids = np.array(unit_ids)
                donors_dict[iv][unit] = {}
                # exclude (target) unit from being included in (donor) unit_ids
                donor_units = (
                    unit_ids[unit_ids != unit] if unit in unit_ids else unit_ids
                )
                num_donors = len(donor_units)

                # get pre-intervention target data
                y1 = self.pre_df.loc[(self.pre_df.unit == unit)]
                y1 = y1.drop(columns=columns).values.reshape(M * T0)

                # get pre-intervention donor data
                X1 = self.pre_df.loc[(self.pre_df.unit.isin(donor_units))]
                donors1 = X1.unit.values
                X1 = X1.drop(columns=columns).values.reshape(num_donors, M * T0).T
                for metric_i, metric in enumerate(metrics):
                    # get post-intervention donor data
                    X2 = self.post_df.loc[
                        (self.post_df.unit.isin(donor_units))
                        & (self.post_df.intervention == iv)
                        & (self.post_df.metric == metric)
                    ]
                    donors = X2.unit.values
                    X2 = X2.drop(columns=columns).values.T
                    # assert np.array_equal(donors,donors1)
                    # print(X2.shape, X1.shape)
                    # make counterfactual predictions
                    yh, beta, std = hsvt_ols(
                        X1,
                        X2,
                        y1,
                        T0=T0,
                        metric_i=metric_i,
                        t=self.t,
                        center=self.center,
                        rcond=self.rcond,
                        include_pre=self.include_pre,
                        method=self.rank_method,
                        return_coefficients=True,
                        use_lasso=self.use_lasso,
                        use_ridge=self.use_ridge,
                    )
                    donors_dict[iv][unit][metric] = dict(zip(donors, beta))
                    # append data
                    yh_data = np.vstack([yh_data, yh]) if yh_data.size else yh
                    std_data = np.vstack([std_data, std]) if std_data.size else std
                    idx_data = (
                        np.vstack(
                            [idx_data, np.array([unit, iv, metric], dtype=object)]
                        )
                        if idx_data.size
                        else np.array([unit, iv, metric], dtype=object)
                    )

        pre_cols = list(self.pre_df.drop(columns=columns).columns)
        post_cols = list(self.post_df.drop(columns=columns).columns)
        df_columns = pre_cols + post_cols if self.include_pre else post_cols
        df_synth = pd.DataFrame(columns=df_columns, data=yh_data)
        df_synth.insert(0, "metric", idx_data[:, 2])
        df_synth.insert(0, "intervention", idx_data[:, 1])
        df_synth.insert(0, "unit", idx_data[:, 0])

        df_std = pd.DataFrame(columns=df_columns, data=std_data)
        df_std.insert(0, "metric", idx_data[:, 2])
        df_std.insert(0, "intervention", idx_data[:, 1])
        df_std.insert(0, "unit", idx_data[:, 0])

        self.synthetic_results = df_synth
        self.std_df = df_std
        self.donors_info = donors_dict

        # Evalutaion
        self.diagnose(self.pre_df, self.post_df, interventions)
        self.get_scores()

        # mSSA CI
        if mSSA_CI:
            # get time series
            #################### FIX, 21 is not the right number. get the post df numbers ####################
            ts = df_synth.loc[:, range(21)].values.T
            ts_diff = ts
            ## mSSA parameters
            model = mSSA(5, rank=2, normalize=True)
            model.fit(ts_diff, train_points=T - T0)
            var_model = mSSA(5, rank=1, normalize=True)
            var_model.fit(
                np.square(ts[: model.imputed.shape[0], :] - model.imputed),
                train_points=T - T0,
            )
            std = np.sqrt(var_model.imputed)
            std = std.T
            std = np.concatenate([np.zeros([std.shape[0], len(pre_cols)]), std], axis=1)
            df_std_mSSA = pd.DataFrame(columns=pre_cols + post_cols[:-1], data=std)
            df_std_mSSA.insert(0, "metric", idx_data[:, 2])
            df_std_mSSA.insert(0, "intervention", idx_data[:, 1].astype("int"))
            df_std_mSSA.insert(0, "unit", idx_data[:, 0])
            self.std_df_mSSA = df_std_mSSA

    def plot_predictions(
        self,
        unit,
        CI="LR",
        labels=None,
        title=None,
        xlabel="days",
        ylabel="count",
        c=95,
        cumlative=True,
        true_values=None,
    ):
        if CI == "mSSA":
            std = self.std_df_mSSA
        else:
            std = self.std_df
        df_r = self.synthetic_results
        df = pd.concat([self.pre_df, self.post_df.iloc[:, 4:]], axis=1)
        df["intervention"] = self.post_df["intervention"]
        interventions = df_r.intervention.unique()
        if labels is None:
            labels = ["intervention_level: %s" % i for i in interventions]
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        plt.figure()
        # Plot interventions
        alpha = float(norm.ppf(0.5 + c / 200))

        for I, IV in enumerate(interventions):
            y_hat = (
                df_r[(df_r.unit == unit) & (df_r.intervention == IV)]
                .iloc[0, 3:]
                .astype("float")
            )
            if cumlative:
                y_hat = np.cumsum(y_hat)
            ## Fix cumlative sum with SD
            plt.plot(y_hat, "--", label=labels[I])
            std_i = (
                std[(df_r.unit == unit) & (df_r.intervention == IV)]
                .iloc[0, 3:]
                .astype("float")
            )
            ub = alpha * std_i + y_hat
            lb = -alpha * std_i + y_hat
            # ub[np.isnan(ub)] = 0; lb[np.isnan(lb)] = 0;
            plt.fill_between(
                df_r.columns[3:].astype("float"), lb, ub, alpha=0.2, color=colors[I]
            )
        # Plot real
        if cumlative:
            df.iloc[:, 4:] = np.cumsum(df.iloc[:, 4:].values, axis=1)
        loc_intervention = int(
            df[(df.unit == unit)].intervention.values[0] - min(interventions)
        )
        if true_values is None:
            plt.plot(
                (df[(df.unit == unit)].iloc[0, 4:].astype("float")),
                color=colors[loc_intervention],
                label="True values (%s)" % labels[loc_intervention],
            )
        else:
            plt.plot(
                true_values,
                color=colors[loc_intervention],
                label="True values (%s)" % labels[loc_intervention],
            )
        plt.axvline(
            int(self.pre_df.columns[-1] + 1),
            linestyle="dashed",
            label="Pre-intervention/Post-intervention split",
            color="k",
        )
        if title is None:
            title = unit
        plt.title(title)
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(title.split("/")[0] + ".png")
        plt.show()

    def get_scores(self):
        if self.synthetic_results is None:
            raise Exception("You need to call self.fit before computing the score ")

        # columns to be dropped
        columns = ["intervention", "unit", "metric"]
        post_df_columns = list(columns)
        if "donor" in self.post_df.columns:
            post_df_columns = post_df_columns + ["donor"]
        ##### CROSS VALIDATION SCORE  ######
        self.cross_validation_score = self._get_score(
            self.synthetic_results, self.post_df, columns, post_df_columns
        )

        ##### Training SCORE  ######
        self.training_score = self._get_score(
            self.synthetic_results, self.pre_df, columns, post_df_columns
        )

    def _get_score(self, synth_df, true_df, columns, post_df_columns):
        units = np.unique(true_df.unit.values)
        metrics = np.unique(true_df.metric.values)
        N = len(units)
        M = len(metrics)
        diag = pd.DataFrame()
        df = synth_df
        R2 = np.zeros(N * M)
        R_rct = np.zeros(N * M)
        units_array = []
        metrics_array = []
        i = 0
        for _, unit in enumerate(units):
            for j, metric in enumerate(metrics):
                ivs = true_df.loc[
                    (true_df.unit == unit) & (true_df.metric == metric), "intervention"
                ].values
                assert len(ivs) == 1
                iv = ivs[0]
                baseline_error_sum = 0
                estimated_error_sum = 0
                y = (
                    true_df.loc[
                        (true_df.unit == unit)
                        & (true_df.intervention == iv)
                        & (true_df.metric == metric)
                    ]
                    .drop(columns=post_df_columns)
                    .values
                )
                y_other_donors = (
                    true_df.loc[
                        (true_df.unit != unit)
                        & (~true_df.unit.isin(self.non_donor_list))
                        & (true_df.intervention == iv)
                        & (true_df.metric == metric)
                    ]
                    .drop(columns=post_df_columns)
                    .values
                )
                # y_hat = df.loc[(df.unit==unit) & (df.intervention==iv)].drop(columns=columns).values
                y_hat = synth_df.loc[
                    (synth_df.unit == unit)
                    & (synth_df.intervention == iv)
                    & (synth_df.metric == metric),
                    true_df.drop(columns=post_df_columns).columns,
                ].values
                R2[i] = self._get_r2(y, y_hat)
                R_rct[i] = self.get_r_rct(y_other_donors, y, y_hat)
                metrics_array.append(metric)
                units_array.append(unit)
                i += 1
        diag.insert(0, "R2 scores", R2)
        diag.insert(0, "R2_rct scores", R_rct)
        diag.insert(0, "unit", units_array)
        diag.insert(0, "metric", metrics_array)
        return diag

    def _get_r2(self, y, y_hat):
        baseline_error_sum = ((y.mean(axis=1) - y) ** 2).sum()
        estimated_error_sum = ((y_hat - y) ** 2).sum()
        return 1 - estimated_error_sum / baseline_error_sum

    def get_r_rct(self, y_other_donors, y, y_hat):
        y_mean = y_other_donors.mean(0)
        baseline_error_sum = ((y_mean - y) ** 2).sum()
        estimated_error_sum = ((y_hat - y) ** 2).sum()
        return 1 - estimated_error_sum / baseline_error_sum
