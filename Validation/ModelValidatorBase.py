import os

import pandas as pd
from sklearn.metrics import mean_squared_error
from zope.interface import implementer
from sklearn.model_selection import LeaveOneGroupOut
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from windrose import WindroseAxes

from Models.IModel import IModel
from Validation.IModelValidator import IModelValidator


@implementer(IModelValidator)
class ModelValidatorBase:
    """
    Base class for model validation tasks.

    This class implements methods for performing various validation tasks on machine learning models,
    such as cross-validation, evaluation of RMSE metrics, and generating RMSE plots.

    It implements the IModelValidator interface.

    Attributes:
        None
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

        # Extract data and labels from the DataFrame
        data = data_frame.drop(columns=['labels'])
        labels = data_frame[['labels', 'stationId']]

        # Group data by stationId for Leave-One-Group-Out (LOGO) cross-validation
        GRP = data_frame['stationId'].tolist()
        logo = LeaveOneGroupOut()

        # Create a copy of the original DataFrame
        data_frame_with_prediction: pd.DataFrame = data_frame.copy()
        # Add a new column named 'prediction' filled with None values
        data_frame_with_prediction['prediction'] = None
        # Get the index of the 'prediction' column in the DataFrame
        col_index = data_frame_with_prediction.columns.get_loc('prediction')

        # Perform cross-validation
        for train_index, test_index in logo.split(data, labels, groups=GRP):
            # Split data into training and testing sets
            data_train, data_test = data.iloc[train_index].copy(), data.iloc[test_index].copy()
            labels_train, labels_test = labels.iloc[train_index].copy(), labels.iloc[test_index].copy()

            # Drop stationId from the data
            data_train.drop(columns=['stationId'], inplace=True)
            data_test.drop(columns=['stationId'], inplace=True)
            labels_train.drop(columns=['stationId'], inplace=True)
            labels_test.drop(columns=['stationId'], inplace=True)

            # Transform data and labels
            data_train_scaled, data_test_scaled = model.data_transform(data_train, data_test)
            labels_train_reval, labels_test_reval = model.label_transform(labels_train, labels_test)

            # Reset and train the model
            model.reset_model()
            model.train(data_train_scaled, labels_train_reval, True)

            # Evaluate predictions
            labels_pred_test = model.evaluate(data_test_scaled)

            # Append predictions to the data_frame_with_prediction dataFrame
            data_frame_with_prediction.iloc[test_index,col_index] = labels_pred_test

        return data_frame_with_prediction

    def RMSE_per_station(self, output_path: str, data_with_prediction: pd.DataFrame) -> None:
        """
        Calculate and save RMSE (Root Mean Squared Error) per station based on the given data with predictions.

        Parameters:
            output_path (str): Path to save the RMSE per station plot.
            data_with_prediction (pd.DataFrame): DataFrame containing data and corresponding predictions.
        """
        feature_name = 'stationId'
        self.__RMSE_per_feature(output_path, data_with_prediction, feature_name, True)

    def RMSE_per_doy(self, output_path: str, data_with_prediction: pd.DataFrame) -> None:
        """
        Calculate and save RMSE (Root Mean Squared Error) per day of year (DOY) based on the given data with predictions.

        Parameters:
            output_path (str): Path to save the RMSE per DOY plot.
            data_with_prediction (pd.DataFrame): DataFrame containing data and corresponding predictions.
        """
        # Extract DOY (Day of Year) from the datetime column and new column.
        feature_name = 'DOY'
        data_with_doy = data_with_prediction.copy()
        if not feature_name in data_with_doy:
            data_with_doy['DOY'] = data_with_prediction['datetime'].dt.day_of_year
        self.__RMSE_per_feature(output_path, data_with_doy, feature_name)

    def RMSE_per_tod(self, output_path: str, data_with_prediction: pd.DataFrame) -> None:
        """
        Calculate and save RMSE (Root Mean Squared Error) per time of day (TOD) based on the given data with predictions.

        Parameters:
            output_path (str): Path to save the RMSE per TOD plot.
            data_with_prediction (pd.DataFrame): DataFrame containing data and corresponding predictions.
        """
        # Extract TOD (Time of Day) from the datetime column and add new column
        feature_name = 'TOD'
        data_with_tod = data_with_prediction.copy()
        data_with_tod['TOD'] = data_with_prediction['datetime'].dt.hour
        self.__RMSE_per_feature(output_path, data_with_tod, feature_name)

    def RMSE_per_wd(self, output_path: str, data_with_prediction: pd.DataFrame, resolution: float) -> None:
        """
        Calculate and save RMSE (Root Mean Squared Error) per wind direction (WD) based on the given data with predictions.

        Parameters:
            output_path (str): Path to save the RMSE per WD plot.
            data_with_prediction (pd.DataFrame): DataFrame containing data and corresponding predictions.
            resolution (float): Resolution for binning wind direction values.
        """

        # Preprocess wind direction data and calculate RMSE per wind direction
        data_with_prediction_wd = data_with_prediction.copy()
        data_with_prediction_wd['Wind_Direction'] = data_with_prediction_wd['Wind_Direction'].apply(
            lambda x: x + 360 if x < 0 else x)

        bins = list(range(0, 361, resolution))

        # Add a new column to your DataFrame with the wind direction range labels
        data_with_prediction_wd['wind_direction_range'] = pd.cut(data_with_prediction_wd['Wind_Direction'], bins=bins,
                                                                 labels=bins[:-1])

        # Group by 'wind_direction_range'
        # Define a lambda function to calculate RMSE if group size is greater than 1, otherwise return None
        wind_RMSE = data_with_prediction_wd.groupby('wind_direction_range', observed=False). \
            apply(lambda g: mean_squared_error(g['labels'], g['prediction'], squared=False) if len(g) > 1 else None)
        # Drop rows with None values
        wind_RMSE.dropna(inplace=True)

        wind_direction = wind_RMSE.index

        # Create a WindroseAxes instance
        ax = WindroseAxes.from_ax()

        # Plot the wind rose using wind direction and wind speed
        ax.bar(wind_direction, wind_RMSE, edgecolor='white')

        # Set title and labels
        ax.set_title('Wind Rose Plot')
        ax.set_legend()

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        # Show the wind rose plot
        plt.savefig(
            os.path.join(output_path, 'windrose_RMSE_LOGO_per_wind_direction.jpg'))
        plt.close()

    def __RMSE_per_feature(self, output_path: str, data_with_prediction: pd.DataFrame, feature_name: str,
                           is_bar_graph: bool = False) -> None:
        """
        Calculate and save RMSE (Root Mean Squared Error) per specified feature.

        Parameters:
            output_path (str): Path to save the RMSE plot.
            data_with_prediction (pd.DataFrame): DataFrame containing data and corresponding prediction.
            feature_name (str): Name of the feature for which RMSE is calculated.
            is_bar_graph (bool): Whether to plot the RMSE values as a bar graph (True) or a line plot (False).
                                                   Defaults to False.
        """
        # Calculate RMSE per specified feature
        feature_RMSE = data_with_prediction.groupby(feature_name, observed=False).apply(
            lambda g: mean_squared_error(g['labels'], g['prediction'], squared=False))
        feature_values = feature_RMSE.index

        # Create plot
        fig, ax = plt.subplots(figsize=(15, 7))

        if is_bar_graph:
            # Plot RMSE as a bar graph
            indices = np.arange(len(feature_values))
            ax.bar(indices, feature_RMSE, color='darkgreen',
                   label='RMSE per {}'.format(feature_name), align='center')
            plt.xticks(rotation=90)
            ax.set_xticks(indices, feature_values)

        else:
            # Plot RMSE as a line plot
            ax.plot(feature_values, feature_RMSE, marker='o', markersize=6, color='darkgreen',
                    label='RMSE per {}'.format(feature_name), linewidth=3)

        # Plot line plot to the mean value of the RMSE (feature_RMSE)
        ax.axhline(y=mean(feature_RMSE), c="red", linewidth=2, zorder=0, label='Mean RMSE')

        # Set plot labels and properties
        ax.set_xlabel(feature_name, fontsize=17)
        ax.set_ylabel('RMSE', fontsize=17)
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)
        fig.patch.set_facecolor('lightgray')
        plt.grid(axis='y')
        plt.tight_layout()
        plt.legend(loc='best', fontsize=15)

        # Save the plot
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        plt.savefig(os.path.join(output_path, 'RMSE_LOGO_per_{}.jpg'.format(feature_name)))
        plt.close()

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
        label_pred = model.evaluate(data_test)
        RMSE = mean_squared_error(label_test, label_pred, squared=False)
        return RMSE
