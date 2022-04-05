import numpy as np
import tensorflow as tf
from glob import glob
from models.handGestureModel import HandGestureModel


def train_gesture():
    print("training")
    hg_model = HandGestureModel()
    x_train, y_train = load_data(hg_model.gesture_types)

    ## start training
    indices = tf.range(start=0, limit=tf.shape(x_train)[0], dtype=tf.int32)
    for _ in range(10):
        idx = tf.random.shuffle(indices)
        x = tf.gather(x_train, idx)
        y = tf.gather(y_train, idx)

        hg_model.fit(x, y)

        x_test, y_test = get_test_data(x_train, y_train)
        _, acc = hg_model.model.evaluate(x_test, y_test, verbose=2)

        if acc >= 1.0:
            break

    ## save
    hg_model.model.save("hand_gesture_model")

    print("done")


def load_data(gesture_types):
    x_train = []
    y_train = []

    for iy, gesture_group in enumerate(gesture_types):
        for gesture in glob(f"./gestures/{gesture_group}/*.txt"):
            g = np.loadtxt(gesture)
            x_train.append(g)
            y_train.append(iy)

    return np.array(x_train, dtype=np.float64), np.array(y_train, dtype=np.int0)


def get_test_data(x_train: np.ndarray, y_train: np.ndarray, sample_size=400):
    random_ix = np.random.randint(0, len(x_train), sample_size)

    x_test = []
    y_test = []

    for ix in random_ix:
        x_test.append(x_train[ix])
        y_test.append(y_train[ix])

    return np.array(x_test, dtype=np.float64), np.array(y_test, dtype=np.int0)
