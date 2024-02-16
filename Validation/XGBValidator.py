import os
import numpy as np
from typing import Dict
import matplotlib.pyplot as plt
import xgboost
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

from Validation.ModelValidatorBase import ModelValidatorBase


class XGBValidator(ModelValidatorBase):
    """
    Implementation of ModelValidatorBase for XGBoost models.

    Inherits from ModelValidatorBase.

    Attributes:
        Inherits all attributes from ModelValidatorBase.

    Methods:
        fine_tuning: Performs hyperparameter tuning using GridSearchCV for an XGBoost model.
        feature_importance: Plots the feature importance of an XGBoost model.

    """

    def fine_tuning(self, data_train: np.ndarray, label_train: np.ndarray, params_dict: Dict[str, list[float]]) -> Dict[
        str, float]:
        """
        Perform hyperparameter fine-tuning using GridSearchCV.

        Parameters:
            data_train (np.ndarray): Training data.
            label_train (np.ndarray): Training labels.
            params_dict (Dict[str, list[float]]): Dictionary containing lists of hyperparameters.

        Returns:
            Dict[str, float]: Dictionary containing the best hyperparameters found.
        """
        max_depth = params_dict['max_depth']
        gamma = params_dict['gamma']
        learning_rate = params_dict['learning_rate']

        # Initialize XGBoost Regressor
        XG = XGBRegressor()

        # Create a pipeline with XGBoost Regressor
        pipe = Pipeline(steps=[('XG', XG)])

        # Define hyperparameter grid
        parameters = dict(XG__learning_rate=learning_rate,
                          XG__max_depth=max_depth,
                          XG__gamma=gamma
                          )

        # Initialize GridSearchCV with the pipeline and hyperparameters
        XG_boost = GridSearchCV(pipe, parameters)

        # Fit the GridSearchCV object to find the best hyperparameters
        XG_boost.fit(data_train, label_train)

        # Extract the best hyperparameters found
        return {'max_depth': XG_boost.best_estimator_.get_params()['XG__max_depth'],
                'gamma': XG_boost.best_estimator_.get_params()['XG__gamma'],
                'learning_rate': XG_boost.best_estimator_.get_params()['XG__learning_rate']}

    def feature_importance(self, xgb_model: XGBRegressor, output_path: str, features_name: list[str]) -> None:
        """
        Plot and save the feature importance of the trained XGBoost model.

        Parameters:
            xgb_model (XGBRegressor): Trained XGBoost model.
            output_path (str): Output path to save the feature importance plot.
        """

        # Create a figure and axis for plotting
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        # Plot feature importance using XGBoost's built-in plot_importance function
        xgboost.plot_importance(xgb_model, importance_type='weight', show_values=True, ax=ax, color='olive',
                                height=0.7, grid=False, title=None)

        # Set yticklabels
        yticklabels = self.__create_yticklabels(ax.get_yticklabels(), features_name)
        ax.set_yticklabels(yticklabels)

        # Set axis labels and title
        ax.set_ylabel(ylabel='features', fontsize=22)
        ax.set_xlabel(xlabel='F Score', fontsize=22)
        ax.set_title('Feature importance - XGboost Regressor', fontsize=22)
        ax.tick_params(axis='both', labelsize=10)

        # Save the plot to the output path
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        plt.savefig(os.path.join(output_path, 'Feature_importance_XGboost.jpg'))
        plt.close()

    def __create_yticklabels(self, current_yticklabels: list[str], features_name: list[str]):
        """
         Create y-axis tick labels with customized feature names.

         Args:
             current_yticklabels (list[str]): Current y-axis tick labels.
             features_name (list[str]): List of feature names.

         Returns:
             list[str]: List of y-axis tick labels with customized feature names.
         """
        # Extract the text from the current y-axis tick labels
        yticklabels = [label.get_text() for label in current_yticklabels]

        # Dictionary mapping original values to new names
        rename_map = {}
        for i in range(len(yticklabels)):
            rename_map[f'f{i}'] = features_name[i]

        # Create a new list with renamed values
        renamed_yticklabels = [rename_map[label] if label in rename_map else label for label in yticklabels]
        return renamed_yticklabels
