print("aaaa")
import tensorflow as tf

from utils.constants import gesture_types


class HandGestureModel:
    def __init__(self):
        self.model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=(21, 2)),
                tf.keras.layers.Dense(80, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(80, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(len(gesture_types)),
            ]
        )

        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.model.compile(optimizer="adam", loss=self.loss_fn, metrics=["accuracy"])

        self.softmax = tf.keras.layers.Softmax()

        self.gesture_types = gesture_types

    def fit(self, x, y, epochs=100, verbose=2):
        self.model.fit(x, y, epochs=epochs, verbose=verbose)

    def predict(self, data):
        prediction = self.model(data)
        return self.softmax(prediction)
