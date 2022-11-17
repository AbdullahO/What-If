from abc import ABC, abstractmethod
import pandas as pd


class StrReprBase:
    def __repr__(self):
        return str(self)

    def __str__(self):
        field_list = []
        for (k, v) in sorted(self.__dict__.items()):
            if (v is None) or (isinstance(v, (float, int))):
                field_list.append("%s=%s" % (k, v))
            elif isinstance(v, str):
                field_list.append("%s='%s'" % (k, v))
        return "%s(%s)" % (self.__class__.__name__, ", ".join(field_list))


class WhatIFAlgorithm(ABC, StrReprBase):
    """Abstract class for what if algorithms"""

    # properties

    @abstractmethod
    def fit(
        self,
        df: pd.DataFrame,
        unit_column: str,
        time_column: str,
        metrics: list,
        actions: list,
        covariates: list = None,
    ):
        """take sparse tensor and return a full tensor

        Args:
            df (pd.DataFrame): data dataframe
            unit_column (str): name for the unit column
            time_column (str): name for the time column
            metrics (list): list of names for the metric columns
            actions (list): list of names for the action columns
            covariates (list, optional): list of names for the covariate columns. Defaults to None.
        """

    @abstractmethod
    def query(self, units, time, metric, action, action_range):
        """returns answer to what if query"""

    @abstractmethod
    def diagnostics(self):
        """returns method-specifc diagnostics"""

    @abstractmethod
    def summary(self):
        """returns method-specifc summary"""

    @abstractmethod
    def save(self, path):
        """save trained model"""

    @abstractmethod
    def load(self, path):
        """load model"""
