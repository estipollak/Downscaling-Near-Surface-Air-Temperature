import os
from typing import Dict
import keras
import numpy as np
import pandas as pd

# INFO messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from Models.ModelBase import ModelBase


class NNModel(ModelBase):
    """Neural Network Model implementing ModelBase class.

    This class defines a neural network model for machine learning tasks.
    It inherits from the ModelBase class and implements methods for setting parameters,
    building the model, training, evaluation, resetting the model, and saving the model.

    Attributes:
        __learning_rate (float): Learning rate for the optimizer.
        __nn_model (tf.keras.Sequential): Neural network model architecture.
    """

    __learning_rate: float
    __nn_model: Sequential

    def set_params(self, params_dict: Dict[str, float]) -> None:
        """
        Set the parameters for the model.

        Parameters:
            params_dict (Dict[str, float]): A dictionary containing model parameters.
        """
        self.__learning_rate = params_dict['learning_rate']

    def __root_mean_squared_error(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Calculate the root mean squared error loss.

        Parameters:
            y_true (tf.Tensor): True labels.
            y_pred (tf.Tensor): Predicted labels.

        Returns:
            tf.Tensor: Root mean squared error loss.
        """
        return keras.backend.sqrt(keras.backend.mean(keras.backend.square(y_pred - y_true)))

    def build_model(self) -> None:
        """Build the neural network model."""
        tf.random.set_seed(0)

        # Define model architecture
        hidden_units1 = 200
        hidden_units2 = 200
        hidden_units3 = 150
        hidden_units4 = 50

        self.__nn_model = Sequential([
            Dense(hidden_units1, kernel_initializer='normal', activation=keras.activations.relu),
            Dense(hidden_units2, kernel_initializer='normal', activation=keras.activations.relu),
            Dense(hidden_units3, kernel_initializer='normal', activation=keras.activations.relu),
            Dense(hidden_units4, kernel_initializer='normal', activation=keras.activations.relu),

            Dense(1, activation=keras.activations.linear)
        ])

        # Compile the model
        self.__nn_model.compile(
            loss=self.__root_mean_squared_error,
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.__learning_rate),
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )

    def train(self, data_train: np.ndarray, labels_train: np.ndarray, early_stop: bool = False) -> None:
        """
        Train the neural network model.

        Parameters:
            data_train (np.ndarray): Training data.
            labels_train (np.ndarray): Training labels.
            early_stop (bool, optional): Whether to use early stopping during training. Defaults to False.
        """



        if early_stop:
            # Configure early stopping callback
            callback = tf.keras.callbacks.EarlyStopping(monitor='val_root_mean_squared_error',
                                                        patience=5,
                                                        restore_best_weights=True,
                                                        mode='auto')

            # Train the model with early stopping
            self.__nn_model.fit(
                data_train,
                labels_train,
                epochs=50,
                batch_size=2000,
                validation_split=0.2,
                callbacks=[callback])

        else:
            # Train the model without early stopping
            self.__nn_model.fit(
                data_train,
                labels_train,
                epochs=500,
                batch_size=2000,
                validation_split=0.2
            )

    def evaluate(self, data: np.ndarray) -> np.ndarray:
        """
        Evaluate the neural network model on the given data.

        Parameters:
            data (np.ndarray): Data to evaluate the model on.

        Returns:
            np.ndarray: Model predictions.
        """
        return self.__nn_model.predict(data).flatten()

    def reset_model(self):
        """Reset the neural network model."""
        keras.backend.clear_session()

    def save_model(self, output_path: str) -> None:
        """
        Save the neural network model architecture and weights to files.

        Parameters:
            output_path (str): The directory where the model files will be saved.
        """
        # Check if the output directory exists, create it if it doesn't
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Save model architecture to JSON file
        model_json = self.__nn_model.to_json()
        with open(os.path.join(output_path, 'nn_model_struct.json'), 'w') as json_file:
            json_file.write(model_json)

        # Save model weights to HDF5 file
        self.__nn_model.save_weights(os.path.join(output_path, 'nn_model_weights.h5'))
