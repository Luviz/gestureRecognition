import numpy as np
import tensorflow as tf
from utils.common import get_configs


configs = get_configs()
gesture_types = configs["gestureTypes"] or []


class HandGestureModel:
    def __init__(self, model_path=None):

        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        if model_path is None:
            self.__build_new_model__()
        else:
            self.model = tf.keras.models.load_model(model_path)
            self.model.summary()

        self.softmax = tf.keras.layers.Softmax()

        self.gesture_types = gesture_types

    def fit(self, x, y, epochs=100, verbose=2):
        self.model.fit(
            x, y, epochs=epochs, verbose=verbose, workers=6, use_multiprocessing=True
        )

    def predict(self, data: np.ndarray):
        prediction = self.model(data)
        return self.softmax(prediction)

    def __build_new_model__(self):
        self.model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=(21, 2)),
                tf.keras.layers.Dense(42, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(42, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(len(gesture_types)),
            ]
        )
        self.model.compile(optimizer="adam", loss=self.loss_fn, metrics=["accuracy"])
