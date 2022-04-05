from os import makedirs
from sys import maxsize
import cv2 as cv
import numpy as np
from utils.common import write_text


from utils.mediapipe.handProcessor import HandProcessor
import utils.constants as constants

handProcessor = HandProcessor(max_num_hands=1)
current_gesture = 0

save_reflected = False


def create_folders():
    for gesture in constants.gesture_types:
        try:
            makedirs(f"gestures/{gesture}")
        except FileExistsError:
            pass


def take(arr: np.ndarray, folder_path="./"):
    values = handProcessor.normalize_coordinates(arr)
    h = hash(values.tobytes()) & maxsize
    path = f"{folder_path}{h:x}.txt"
    np.savetxt(path, values)


def gesture_recoder(cam_src=None):
    create_folders()
    global current_gesture, save_reflected

    if cam_src == None:
        cam_src = 0
    cap = cv.VideoCapture(cam_src)

    c = 0
    key = 0
    run = True
    try:
        while run:
            c = c + 1
            has_frame, frame = cap.read()
            if has_frame:
                h, w, _ = frame.shape
                handProcessor.proc(frame[:, :, ::-1])
                for hand in handProcessor.get_landmark_as_np_arr():
                    for lm in (hand * [w, h]).astype(np.int0):
                        cv.circle(frame, lm, 6, (0, 0, 150), -1)
                        cv.circle(frame, lm, 4, (250, 0, 0), -1)

                txt = f"{constants.gesture_types[current_gesture]} {'*' if save_reflected else ''}"

                write_text(frame, txt, (0 + 10, h - 20))

                cv.imshow("main", frame)

            waitKey = cv.waitKey(10)
            if waitKey > 0:
                gesture_types = constants.gesture_types
                if waitKey in [ord(str(i)) for i in range(10)]:
                    print(f"{waitKey=}, {key}")
                    key = int(chr(waitKey))

                if waitKey == ord(" "):
                    print("take")
                    try:
                        hand = handProcessor.get_landmark_as_np_arr()[0]
                        take(hand, f"gestures/{gesture_types[current_gesture]}/")
                        if save_reflected:
                            take(
                                hand * [-1, 1],
                                f"gestures/{gesture_types[current_gesture]}/",
                            )

                    except IndexError:
                        print("No hand detected!")

                if waitKey == 81 or waitKey == 83:
                    diff = waitKey - 82
                    current_gesture = (current_gesture + diff) % len(gesture_types)

                if waitKey == ord("z"):
                    save_reflected = not save_reflected

                if waitKey == ord("q") or waitKey == 27:
                    run = False

    except KeyboardInterrupt as e:
        print("quiting")
