from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class WhatIFAlgorithm(ABC):
    """Abstract class for what if algorithms"""

    # proerties

    @abstractmethod
    def fit(self, df: pd.DataFrame, labels: dict):
        """take sparse tensor and return a full tensor

        Args:
            df (pd.DataFrame): data dataframe
            labels (dict): labels for unit, time, metric, action, and covaraite

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


class FillTensorBase(ABC):
    """Abstract class for fill_tensor algorithms"""

    # proerties
    @property
    def tensor_shape(self):
        return self._tensor_shape

    @tensor_shape.setter
    def tensor_shape(self, shape_):
        assert isinstance(shape_, tuple)
        self._tensor_shape = shape_

    @abstractmethod
    def fill_tensor(self, Y: np.typing.NDArray[Any]) -> np.typing.NDArray[Any]:
        """take sparse tensor and return a full tensor

        Args:
            Y (np.array): sparse tensor

        Returns:
            np.array: filled tensor
        """

    @abstractmethod
    def update_estimate(
        self, values: np.typing.NDArray[Any], coords: np.typing.NDArray[Any]
    ) -> np.typing.NDArray[Any]:
        """update estimate given new observations

        Args:
            values (np.array): new values (size: (number of new observations,))
            coords (np.array): coordinates of new values  (size: (number of new observations, tensor dimensions))
        Returns:
            np.array: updated filled tensor
        """

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
