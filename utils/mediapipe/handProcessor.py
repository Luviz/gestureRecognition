from dataclasses import dataclass
import mediapipe.python.solutions.hands as mph
import numpy as np

from utils.mediapipe.handProtocol import (
    Classification,
    Point,
    Processed_hands,
    Multi_handedness,
    Hand_landmarks,
)


@dataclass
class HandData:
    landmarks: list[Point]
    classification: Classification
    multi_hand_landmarks: Hand_landmarks


class HandProcessor(mph.Hands):
    _current_process: Processed_hands  # = NamedTuple("Processed_hands", [multi_hand_landmarks, ])

    def proc(self, frame) -> Processed_hands:
        self._current_process = self.process(frame)
        return self._current_process

    def get_handedness(self) -> list[Multi_handedness]:
        return self._current_process.multi_handedness or []

    def get_landmarks(self) -> list[Hand_landmarks]:
        return self._current_process.multi_hand_landmarks or []

    def get_handData(self) -> list[HandData]:
        handedness = self.get_handedness()
        lms = self.get_landmarks()
        return [
            HandData(
                landmarks=lms[ix].landmark,
                multi_hand_landmarks=lms[ix],
                classification=h.classification[0],
            )
            for ix, h in enumerate(handedness)
        ]

    def get_landmark_as_np_arr(self, scale=[1, 1]):
        arr = []
        for lms in self.get_landmarks():
            arr.append([np.array([lm.x, lm.y]) * scale for lm in lms.landmark])

        return np.array(arr)
