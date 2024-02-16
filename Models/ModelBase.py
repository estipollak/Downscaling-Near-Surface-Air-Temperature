import numpy as np
import pandas as pd
import zope
from sklearn.preprocessing import StandardScaler

from Models.IModel import IModel


@zope.interface.implementer(IModel)
class ModelBase:
    """Base class implementing the IModel interface."""

    def data_transform(self, data_train: pd.DataFrame, data_test: pd.DataFrame) -> [np.ndarray, np.ndarray]:
        """
        Transform the input data for training and testing.

        Args:
            data_train (pd.DataFrame): The training data.
            data_test (pd.DataFrame): The testing data.

        Returns:
            [np.ndarray, np.ndarray]: Transformed training and testing data.
        """
        scaler = StandardScaler()
        data_train_scaled = scaler.fit_transform(data_train)
        data_test_scaled = scaler.transform(data_test)
        return data_train_scaled, data_test_scaled

    def label_transform(self, labels_train: pd.DataFrame, labels_test: pd.DataFrame) -> [np.ndarray, np.ndarray]:
        """
        Transform the labels for training and testing.

        Args:
            labels_train (pd.DataFrame): The training labels.
            labels_test (pd.DataFrame): The testing labels.

        Returns:
            [np.ndarray, np.ndarray]: Transformed training and testing labels.
        """
        labels_train_ravel = np.array([labels_train]).ravel()
        labels_test_ravel = np.array([labels_test]).ravel()
        return labels_train_ravel, labels_test_ravel
