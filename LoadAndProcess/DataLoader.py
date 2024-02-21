import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from zope.interface import implementer

from LoadAndProcess.IDataLoader import IDataLoader
from LoadAndProcess.IDataPreprocessor import IDataPreprocessor


@implementer(IDataLoader)
class DataLoader:
    """
    Class responsible for loading and preprocessing data.
    """

    def __init__(self, data_preprocessor: IDataPreprocessor):
        """
        Constructor for DataLoader class.

        Args:
            data_preprocessor (IDataPreprocessor): An object implementing the IDataPreprocessor interface.
                It is responsible for preprocessing the data.
        """
        self.__data_preprocessor = data_preprocessor

    def preprocessing(self, is_hourly_data: bool) -> [pd.DataFrame, pd.DataFrame]:
        """
        Method for data preprocessing.

        Args:
            is_hourly_data (bool): Indicates whether the calculations are on daily/hourly data.

        Returns:
            [pd.DataFrame, pd.DataFrame]: A tuple containing two Pandas DataFrames.
                The first DataFrame represents the Datasets for running,
                and the second DataFrame represents the Datasets for LOGO cross-validation.
        """
        return self.__data_preprocessor.process(is_hourly_data)

    def split_data(self, X: pd.DataFrame, Y: pd.DataFrame, test_size, random_state) -> [np.ndarray, np.ndarray,
                                                                                        np.ndarray, np.ndarray]:
        """
        Method to split data into training and testing sets.

        Args:
            X (pd.DataFrame): Input features.
            Y (pd.DataFrame): Output labels.
            test_size: The proportion of the dataset to include in the test split.
            random_state: Controls the randomness of the split.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing four NumPy arrays.
                The first two arrays represent the training features and labels (X_train, Y_train),
                and the last two arrays represent the testing features and labels (X_test, Y_test).
        """
        return train_test_split(X, Y, test_size=test_size, random_state=random_state)
