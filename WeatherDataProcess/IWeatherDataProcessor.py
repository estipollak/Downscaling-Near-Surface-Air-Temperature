from zope.interface import Interface, Attribute

from LoadAndProcess.IDataLoader import IDataLoader
from LoadAndProcess.IDataPreprocessor import IDataPreprocessor
from Validation.IModelValidator import IModelValidator


class IWeatherDataProcessor(Interface):
    """
    Interface for processing weather data including running machine learning models.
    """

    __data_preprocessor: IDataPreprocessor = Attribute("Preprocessor for weather data")
    __data_loader: IDataLoader = Attribute("Loader for weather data")
    __xgb_validator: IModelValidator = Attribute("Validator for XGBoost models")
    __nn_validator: IModelValidator = Attribute("Validator for neural network models")

    def run_xgb(self, output_path: str, is_best_params: bool = False) -> None:
        """
        Run XGBoost model processing on weather data.

        Args: output_path (str): Path to save output. is_best_params (bool, optional): Whether to use the
        hyperparameter fine-tuning using GridSearchCV to find the best hyperparameters. Defaults to False.
        """
        pass

    def run_nn(self) -> None:
        """
        Run neural network model processing on weather data.
        """
        pass

    def run_logo_nn(self, is_hourly_data: bool, output_path: str) -> None:
        """
        Run Leave-One-Group-Out cross-validation for neural network model on weather data.

        Args:
            is_hourly_data (bool): Indicates whether the calculations are on ERA5 daily/hourly data.
            output_path (str): Path to save output.
        """
        pass

    def run_logo_xgb(self, is_hourly_data: bool, output_path: str) -> None:
        """
        Run Leave-One-Group-Out cross-validation for XGBoost model on weather data.

        Args:
            is_hourly_data (bool): Indicates whether the calculations are on ERA5 daily/hourly data.
            output_path (str): Path to save output.
        """
        pass

    def print_result_of_reference(self) -> None:
        """
        Print results of reference for weather data processing.
        """
        pass
