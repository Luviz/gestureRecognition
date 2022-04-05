import cv2 as cv
import numpy as np

from utils.mediapipe.handProcessor import HandProcessor

handProcessor = HandProcessor()


def gesture_recoder(cam_src=None):
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

                    # print(handProcessor.normalize_coordinates(hand))
                    # for m in handProcessor.normalize_coordinates(hand):
                    #     pta = m * [w, h]
                    #     ptb = (m * [-1, 1] + [1, 0]) * [w, h]

                    #     cv.circle(frame, pta.astype(np.int0), 6, (0, 0, 200), -1)
                    #     cv.circle(frame, ptb.astype(np.int0), 6, (200, 0, 200), -1)

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
