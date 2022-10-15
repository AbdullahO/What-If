# https://github.com/AbdullahO/Synthetic-Interventions
import pandas as pd

import algorithms.base as base

# TODO: copy relevant algorithm from # https://github.com/AbdullahO/Synthetic-Interventions
# 1. Test this algo as a whole
# 2. Refactor and test individual functions
# 3. Consider moving to dask or xarray where appropriate
# https://examples.dask.org/dataframes/03-from-pandas-to-dask.html#Gotcha’s-from-Pandas-to-Dask

class SI(base.WhatIFAlgorithm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # These will raise errors if not found
        self.t = kwargs.pop("t")
        self.center = kwargs.pop("center")

    # TODO: pre_df/post_df in SI -> labels here
    # Work with Abdullah to decide how to handle this
    def fit(self, df: pd.DataFrame, labels: dict):
        """take sparse tensor and return a full tensor

        Args:
            df (pd.DataFrame): data dataframe
            labels (dict): labels for unit, time, metric, action, and covariate

        """

    def query(self, units, time, metric, action, action_range):
        """returns answer to what if query"""
        raise NotImplementedError()

    def diagnostics(self):
        """returns method-specifc diagnostics"""
        raise NotImplementedError()

    def summary(self):
        """returns method-specifc summary"""
        raise NotImplementedError()

    def save(self, path):
        """save trained model"""
        raise NotImplementedError()

    def load(self, path):
        """load model"""
        raise NotImplementedError()
