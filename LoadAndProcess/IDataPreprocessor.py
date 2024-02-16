import pandas as pd
from zope.interface import Interface


class IDataPreprocessor(Interface):
    """
    Interface for classes responsible for preprocessing data.
    """

    def process(self) -> [pd.DataFrame, pd.DataFrame]:
        """
        Method for preprocessing data.

        Returns:
            [pd.DataFrame, pd.DataFrame]: A tuple containing two Pandas DataFrames.
                The first DataFrame represents the Datasets for running,
                and the second DataFrame represents the Datasets for LOGO cross-validation.
        """
        pass


