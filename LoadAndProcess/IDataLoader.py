import numpy as np
import pandas as pd
from zope.interface import Interface, Attribute

from LoadAndProcess.IDataPreprocessor import IDataPreprocessor


class IDataLoader(Interface):
    """
    Interface for classes responsible for loading and preprocessing data.
    """
    __data_preprocessor: IDataPreprocessor = Attribute("An attribute indicating the data preprocessor object")

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
        pass

    def split_data(self, X, Y, test_size, random_state) -> [np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Method for splitting data into training and testing sets.

        Args:
            X: Input features.
            Y: Output labels.
            test_size: The proportion of the dataset to include in the test split.
            random_state: Controls the randomness of the split.

        Returns:
            [np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing four NumPy arrays.
               The first two arrays represent the training features and labels (X_train, Y_train),
               and the last two arrays represent the testing features and labels (X_test, Y_test).
        """
        pass
