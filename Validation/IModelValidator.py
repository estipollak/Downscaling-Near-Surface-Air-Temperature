import pandas as pd
from zope.interface import Interface

from Models.IModel import IModel


class IModelValidator(Interface):
    """
    Interface for model validation tasks.

    This interface defines methods for performing various validation tasks on machine learning models,
    such as cross-validation and evaluation of RMSE (Root Mean Squared Error) metrics.
    """

    def LOGO_cross_validation(self, data_frame: pd.DataFrame, model: IModel) -> pd.DataFrame:
        """
        Perform Leave-One-Group-Out (LOGO) cross-validation on the given data with the specified model.

        Parameters:
            data_frame (pd.DataFrame): DataFrame containing the dataset to be validated.
            model (IModel): Machine learning model to be validated.

        Returns:
            pd.DataFrame: DataFrame containing the prediction results.
        """
        pass

    def RMSE_per_station(self, output_path: str, data_with_prediction: pd.DataFrame) -> None:
        """
        Calculate and save RMSE (Root Mean Squared Error) per station based on the given data with predictions.

        Parameters:
            output_path (str): Path to save the RMSE per station plot.
            data_with_prediction (pd.DataFrame): DataFrame containing data and corresponding predictions.
        """
        pass

    def RMSE_per_doy(self, output_path: str, data_with_prediction: pd.DataFrame) -> None:
        """
        Calculate and save RMSE (Root Mean Squared Error) per day of year (DOY) based on the given data with predictions.

        Parameters:
            output_path (str): Path to save the RMSE per DOY plot.
            data_with_prediction (pd.DataFrame): DataFrame containing data and corresponding predictions.
        """
        pass

    def RMSE_per_tod(self, output_path: str, data_with_prediction: pd.DataFrame) -> None:
        """
        Calculate and save RMSE (Root Mean Squared Error) per time of day (TOD) based on the given data with predictions.

        Parameters:
            output_path (str): Path to save the RMSE per TOD plot.
            data_with_prediction (pd.DataFrame): DataFrame containing data and corresponding predictions.
        """
        pass

    def RMSE_per_wd(self, output_path: str, data_with_prediction: pd.DataFrame, resolution: float) -> None:
        """
        Calculate and save RMSE (Root Mean Squared Error) per wind direction (WD) based on the given data with predictions.

        Parameters:
            output_path (str): Path to save the RMSE per WD plot.
            data_with_prediction (pd.DataFrame): DataFrame containing data and corresponding predictions.
            resolution (float): Resolution for binning wind direction values.
        """
        pass

    def evaluate_RMSE(self, data_test: pd.DataFrame, label_test: pd.DataFrame, model: IModel) -> float:
        """
        Evaluate the RMSE (Root Mean Squared Error) metric on the given test data and labels using the specified model.

        Parameters:
            data_test (pd.DataFrame): DataFrame containing the test data.
            label_test (pd.DataFrame): DataFrame containing the test labels.
            model (IModel): Machine learning model to evaluate.

        Returns:
            float: RMSE value for the model's predictions on the test data.
        """
        pass
