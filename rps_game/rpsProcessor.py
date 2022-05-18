import numpy as np

from utils.mediapipe.handProcessor import HandProcessor
from models.handGestureModel import HandGestureModel


class RPS_Processor:
    def __init__(self):
        self.handProcessor = HandProcessor(max_num_hands=2)
        self.hg_model = HandGestureModel("hand_gesture_model")
        self.handGestureLabel = [
            "paper",
            "rock",
            "unknown",
            "scissors",
            "scissors",
            "unknown",
            "unknown",
            "scissors",
            "scissors",
            "unknown",
            "unknown",
        ]

    def process_img(self, img):
        self.has_hand = False
        self.img = img
        self.height, self.width, self.channel = img.shape
        self.handProcessor.proc(self.img)
        self.hands = self.handProcessor.get_landmark_as_np_arr()
        if len(self.hands) > 0:
            self.has_hand = True
            self._get_bounding_boxes_labels()

    def _get_bounding_boxes_labels(self):
        self.boxes = []
        self.labels = []
        for hand in self.hands:
            pt1 = np.min(hand, axis=0)
            pt2 = np.max(hand, axis=0)
            self.boxes.append([pt1, pt2])
            n_hand = self.handProcessor.normalize_coordinates(hand)
            p = self.hg_model.predict(np.array([n_hand]))
            label = self.handGestureLabel[np.argmax(p[0])]
            self.labels.append(label)
