import pandas as pd
from zope.interface import Interface


class IDataPreprocessor(Interface):
    """
    Interface for classes responsible for preprocessing data.
    """

    def process(self, is_hourly_data: bool) -> [pd.DataFrame, pd.DataFrame]:
        """
        Method for preprocessing data.

        Args:
            is_hourly_data (bool): Indicates whether the calculations are on daily/hourly data.

        Returns:
            [pd.DataFrame, pd.DataFrame]: A tuple containing two Pandas DataFrames.
                The first DataFrame represents the Datasets for running,
                and the second DataFrame represents the Datasets for LOGO cross-validation.
        """
        pass
