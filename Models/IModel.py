from typing import Dict
import numpy as np
import pandas as pd
from zope.interface import Interface


class IModel(Interface):
    """
    Interface for machine learning models.
    This interface defines methods for transforming data, training models, evaluating performance,
    and saving models to disk.
    """

    def data_transform(self, data_train: pd.DataFrame, data_test: pd.DataFrame) -> [np.ndarray, np.ndarray]:
        """
        Transform the input data for training and testing.

        Args:
            data_train (pd.DataFrame): The training data.
            data_test (pd.DataFrame): The testing data.

        Returns:
            [np.ndarray, np.ndarray]: Transformed training and testing data.
        """
        pass

    def label_transform(self, labels_train: pd.DataFrame, labels_test: pd.DataFrame) -> [np.ndarray, np.ndarray]:
        """
       Transform the labels for training and testing.

       Args:
           labels_train (pd.DataFrame): The training labels.
           labels_test (pd.DataFrame): The testing labels.

       Returns:
           [np.ndarray, np.ndarray]: Transformed training and testing labels.
       """
        pass

    def set_params(self, params_dict: Dict[str, float]) -> None:
        """
        Set the parameters for the model.

        Args:
            params_dict (Dict[str, float]): A dictionary containing model parameters.
        """
        pass

    def build_model(self) -> None:
        """Build the model."""
        pass

    def train(self, data_train: np.ndarray, labels_train: np.ndarray, early_stop: bool = False) -> None:
        """
        Train the model.

        Args:
            data_train (np.ndarray): The training data.
            labels_train (np.ndarray): The training labels.
            early_stop (bool, optional): Whether to use early stopping during training. Defaults to False.
        """
        pass

    def evaluate(self, data: np.ndarray) -> np.ndarray:
        """
        Evaluate the model on the given data.

        Args:
            data (np.ndarray): The data to evaluate the model on.

        Returns:
            np.ndarray: The model predictions.
        """
        pass

    def reset_model(self):
        """Reset the model."""
        pass

    def save_model(self, output_path: str) -> None:
        """
        Save the model to a file.

        Args:
            output_path (str): The path to save the model.
        """
        pass
