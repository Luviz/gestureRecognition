import cv2 as cv
import numpy as np
from utils.mediapipe.handProcessor import HandProcessor

handProcessor = HandProcessor()


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

                for hand in handProcessor.get_landmark_as_np_arr([w,h]).astype(np.int0):
                    for h_lm in hand:
                        cv.circle(frame, h_lm, 5, (200, 0, 0), -1)

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
