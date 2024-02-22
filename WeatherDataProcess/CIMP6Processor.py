from typing import Dict
import pandas as pd
from sklearn.metrics import mean_squared_error
from zope.interface import implementer

from LoadAndProcess.CIMP6DataPreprocessor import CIMP6DataPreprocessor
from LoadAndProcess.DataLoader import DataLoader
from Models.NNModel import NNModel
from Models.XGBModel import XGBModel
from Validation.ModelValidatorBase import ModelValidatorBase
from Validation.XGBValidator import XGBValidator
from WeatherDataProcess.IWeatherDataProcessor import IWeatherDataProcessor


@implementer(IWeatherDataProcessor)
class CIMP6Processor:
    def __init__(self, is_hourly_data: bool):
        """
        Initializes the CIMP6Processor class.

        Args:
            is_hourly_data (bool): Indicates whether the data is in hourly or daily resolution.
        """
        # Initialize preprocessor, loader, and validators
        self.__data_preprocessor = CIMP6DataPreprocessor()
        self.__data_loader = DataLoader(self.__data_preprocessor)
        self.__xgb_validator = XGBValidator()
        self.__nn_validator = ModelValidatorBase()
        # Prepare datasets for running and leave-one-group-out (LOGO) cross-validation
        self.__dataset_for_run, self.__dataset_for_logo = self.__data_preparation(is_hourly_data)

    def __data_preparation(self,  is_hourly_data: bool) -> [pd.DataFrame, pd.DataFrame]:
        """
        Prepares datasets for running and LOGO cross-validation.

        Args:
            is_hourly_data (bool): Indicates whether the calculations are on ERA5 daily/hourly data.

        Returns:
            [pd.DataFrame, pd.DataFrame]: Datasets for running and LOGO cross-validation.
        """
        dataset_for_run, dataset_for_logo = self.__data_loader.preprocessing(is_hourly_data)
        return dataset_for_run, dataset_for_logo

    def run_xgb(self, output_path: str, is_best_params: bool = False) -> None:
        """
        Runs XGBoost model.

        Args:
             output_path (str): Path to save output.
             is_best_params (bool, optional): Whether to use the
                hyperparameter fine-tuning using GridSearchCV to find the best hyperparameters. Defaults to False.
        """
        # Initialize XGBoost model
        xgb_model = XGBModel()

        # Split data into train and test sets
        dataset_for_run: pd.DataFrame = self.__dataset_for_run.copy()
        X_Train, X_Test, Y_Train, Y_Test = self.__data_loader.split_data(dataset_for_run.drop(columns=['labels']),
                                                                         dataset_for_run[['labels']], 0.2, 1)
        # Transform data
        data_train, data_test = xgb_model.data_transform(X_Train, X_Test)
        label_train, label_test = xgb_model.label_transform(Y_Train, Y_Test)

        if is_best_params:
            # seeks to find the best combination of hyperparameters from a predefined set of options
            best_params: Dict[str, float] = self.__xgb_validator.fine_tuning(data_train, label_train,
                                                                             {'learning_rate': [0.5, 0.7, 1],
                                                                              'max_depth': [3, 6, 10],
                                                                              'gamma': [0.1, 0.3, 0.6]})
            # Update the XGBoost model (xgb_model) with the best hyperparameters obtained from the tuning process
            xgb_model.set_params(
                {'learning_rate': best_params['learning_rate'], 'max_depth': best_params['max_depth'],
                 'gamma': best_params['gamma']})
        else:
            # default hyperparameters used
            xgb_model.set_params({'learning_rate': 0.5, 'max_depth': 6, 'gamma': 0.1})

        # Build and train model
        xgb_model.build_model()
        xgb_model.train(data_train, label_train)

        # Evaluate model
        self.__xgb_validator.feature_importance(xgb_model.get_model(), output_path, X_Train.columns.tolist())
        RMSE = self.__nn_validator.evaluate_RMSE(data_test, label_test, xgb_model)
        print('XGB - RMSE on test set is: ', RMSE)

    def run_nn(self) -> None:
        """
        Runs Neural Network (NN) model.
        """
        # Initialize NN model
        nn_model = NNModel()

        # Split data into train and test sets
        dataset_for_run: pd.DataFrame = self.__dataset_for_run.copy()
        X_Train, X_Test, Y_Train, Y_Test = self.__data_loader.split_data(dataset_for_run.drop(columns=['labels']),
                                                                         dataset_for_run[['labels']], 0.2, 1)
        # Transform data
        data_train, data_test = nn_model.data_transform(X_Train, X_Test)
        label_train, label_test = nn_model.label_transform(Y_Train, Y_Test)

        # Set hyperparameters and build model
        nn_model.set_params({'learning_rate': 0.01})
        nn_model.build_model()

        # Train model
        nn_model.train(data_train, label_train)

        # Evaluate model
        RMSE = self.__nn_validator.evaluate_RMSE(data_test, label_test, nn_model)
        print('NN - RMSE on test set is: ', RMSE)

    def run_logo_nn(self, is_hourly_data: bool, output_path: str) -> None:
        """
        Runs Leave-One-Group-Out (LOGO) cross-validation for Neural Network (NN) model.

        Args:
            is_hourly_data (bool): Indicates whether the calculations are on ERA5 daily/hourly data.
            output_path (str): Path to save output.
        """

        # Prepare dataset for LOGO cross-validation
        dataset_for_logo: pd.DataFrame = self.__dataset_for_logo.copy()

        nn_model_for_validation = NNModel()
        nn_model_for_validation.set_params({'learning_rate': 0.01})
        nn_model_for_validation.build_model()

        # Perform LOGO cross-validation
        nn_prediction: pd.DataFrame = self.__nn_validator.LOGO_cross_validation(dataset_for_logo,
                                                                                nn_model_for_validation)
        nn_prediction.reset_index(inplace=True)

        # Calculate and plot graphs representing RMSE for each station and DOY.
        self.__nn_validator.RMSE_per_station(output_path, nn_prediction)
        self.__nn_validator.RMSE_per_doy(output_path, nn_prediction)
        # If the data is in hourly resolution, calculate the RMSE per time of day (TOD)
        if is_hourly_data:
            self.__nn_validator.RMSE_per_tod(output_path, nn_prediction)
        # Calculate and print total RMSE for the entire cross-validation
        print('Total logo nn RMSE: ' + str(
            mean_squared_error(nn_prediction['labels'], nn_prediction['prediction'], squared=False)))

    def run_logo_xgb(self, is_hourly_data: bool, output_path: str) -> None:
        """
        Runs Leave-One-Group-Out (LOGO) cross-validation for XGBoost (XGB) model.

        Args:
            is_hourly_data (bool): Indicates whether the calculations are on ERA5 daily/hourly data.
            output_path (str): Path to save output.
        """
        # Prepare dataset for LOGO cross-validation
        dataset_for_logo: pd.DataFrame = self.__dataset_for_logo.copy()

        # Initialize XGB model for validation
        xgb_model_for_validation = XGBModel()
        xgb_model_for_validation.set_params({'learning_rate': 0.5, 'max_depth': 6, 'gamma': 0.1})
        xgb_model_for_validation.build_model()

        # Perform LOGO cross-validation
        xgb_prediction: pd.DataFrame = self.__xgb_validator.LOGO_cross_validation(dataset_for_logo,
                                                                                  xgb_model_for_validation)
        xgb_prediction.reset_index(inplace=True)

        # Calculate and plot graphs representing RMSE for each station and DOY.
        self.__xgb_validator.RMSE_per_station(output_path, xgb_prediction)
        self.__xgb_validator.RMSE_per_doy(output_path, xgb_prediction)
        # If the data is in hourly resolution, calculate the RMSE per time of day (TOD)
        if is_hourly_data:
            self.__xgb_validator.RMSE_per_tod(output_path, xgb_prediction)
        # Calculate and print total RMSE for the entire cross-validation
        print('Total logo xgb RMSE: ' + str(
            mean_squared_error(xgb_prediction['labels'], xgb_prediction['prediction'], squared=False)))
