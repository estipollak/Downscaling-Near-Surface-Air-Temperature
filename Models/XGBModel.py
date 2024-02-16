import os
from typing import Dict

import numpy as np
from xgboost import XGBRegressor

from Models.ModelBase import ModelBase


class XGBModel(ModelBase):
    """XGBoost Model implementing ModelBase class.

    This class defines a model based on XGBoost regressor for machine learning tasks.
    It inherits from the ModelBase class and implements methods for setting parameters,
    building the model, training, evaluation, resetting the model, saving the model,
    and retrieving the trained model.

    Attributes:
        __max_depth (int): Maximum tree depth for the XGBoost model.
        __gamma (float): Specifies the minimum loss reduction required to make a split.
        __learning_rate (float): Learning rate for the XGBoost model.
        __xgb_model (XGBRegressor): XGBoost regressor model instance.
    """
    __max_depth: int
    __gamma: float
    __learning_rate: float
    __xgb_model: XGBRegressor

    def set_params(self, params_dict: Dict[str, float]) -> None:
        """
        Set the parameters for the model.

        Parameters:
            params_dict (Dict[str, float]): A dictionary containing model parameters.
        """
        self.__max_depth = int(params_dict['max_depth'])
        self.__gamma = params_dict['gamma']
        self.__learning_rate = params_dict['learning_rate']

    def build_model(self) -> None:
        """Build the XGBoost model."""
        self.__xgb_model = XGBRegressor(verbosity=0, randon_state=1, learning_rate=self.__learning_rate,
                                        max_depth=self.__max_depth, gamma=self.__gamma, metrics='RMSE')

    def train(self, data_train: np.ndarray, labels_train: np.ndarray, early_stop: bool = False) -> None:
        """
        Train the XGBoost model.

        Parameters:
            data_train (np.ndarray): Training data.
            labels_train (np.ndarray): Training labels.
            early_stop (bool, optional): Whether to use early stopping during training. Defaults to False.
        """
        if early_stop:
            # Train the model without early stopping
            self.__xgb_model.fit(data_train, labels_train)
        else:
            # Train the model with early stopping
            self.__xgb_model = XGBRegressor(verbosity=0, randon_state=1, learning_rate=self.__learning_rate,
                                            max_depth=self.__max_depth, gamma=self.__gamma, metrics='RMSE',
                                            n_estimators=500, nfold=5)
            self.__xgb_model.fit(data_train, labels_train)

    def evaluate(self, data: np.ndarray) -> np.ndarray:
        """
        Evaluate the XGBoost model on the given data.

        Parameters:
            data (np.ndarray): Data to evaluate the model on.

        Returns:
            np.ndarray: Model predictions.
        """
        return self.__xgb_model.predict(data)

    def reset_model(self):
        """Reset the XGBoost model."""
        self.build_model()

    def save_model(self, output_path: str) -> None:
        """
        Save the XGBoost model to a file.

        Parameters:
            output_path (str): The directory where the model file will be saved.
        """
        # Check if the output directory exists, create it if it doesn't
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Save model to JSON file
        self.__xgb_model.save_model(os.path.join(output_path, 'xgb_model.json'))

    def get_model(self) -> XGBRegressor:
        """
        Retrieve the trained XGBoost model.

        Returns:
            XGBRegressor: Trained XGBoost model instance.
        """
        return self.__xgb_model
