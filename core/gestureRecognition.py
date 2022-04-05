import cv2 as cv
import numpy as np
from utils.common import write_text
from utils.mediapipe.handProcessor import HandProcessor
from models.handGestureModel import HandGestureModel

handProcessor = HandProcessor(max_num_hands=1)
hg_model = HandGestureModel("hand_gesture_model")


def gesture_recognition(cam_src=None):
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
                    n_hand = handProcessor.normalize_coordinates(hand)
                    p = hg_model.predict(np.array([n_hand]))
                    ix = np.argmax(p[0])
                    txt = f"{hg_model.gesture_types[ix]}:{int(p[0][ix]* 100)}%"
                    for h_lm in (hand * [w, h]).astype(np.int0):
                        cv.circle(frame, h_lm, 5, (200, 0, 0), -1)

                    pt = np.average(hand * [w, h], axis=0).astype(np.int0)
                    write_text(frame, txt, pt)

                cv.imshow("main", frame)

            waitKey = cv.waitKey(10)
            if waitKey > 0:
                # print(waitKey, chr(waitKey))
                if waitKey in [ord(str(i)) for i in range(10)]:
                    print(f"{waitKey=}, {key}")
                    key = int(chr(waitKey))
                if waitKey == ord("q") or waitKey == 27:
                    run = False

    except KeyboardInterrupt as e:
        print("quiting")
