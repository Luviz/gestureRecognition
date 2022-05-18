import cv2 as cv
import numpy as np
from utils.common import write_text

from rps_game.rpsProcessor import RPS_Processor


def get_victor(labels: list[str]):
    if labels[0] == labels[1]:
        return -1
    if "unknown" in labels:
        return -2
    ## rock     -> scissors
    if labels[0] == "rock":
        return 0 if labels[1] == "scissors" else 1
    ## scissors -> paper
    if labels[0] == "scissors":
        return 0 if labels[1] == "paper" else 1
    ## paper    -> rock
    if labels[0] == "paper":
        return 0 if labels[1] == "rock" else 1


def main(cam_src=None):
    if cam_src == None:
        cam_src = 0
    cap = cv.VideoCapture(cam_src)

    c = 0
    key = 0
    run = True
    try:
        rsp_proc = RPS_Processor()
        while run:
            c = c + 1
            has_frame, frame = cap.read()
            if has_frame:
                h, w, _ = frame.shape

                game_view = frame[:, ::-1, ::-1].copy()
                user_view = frame[:, ::-1, :].copy()

                rsp_proc.process_img(game_view)
                colors = [(200, 0, 0), (0, 0, 200)]
                if rsp_proc.has_hand and len(rsp_proc.hands) == 2:
                    dims = np.array([w, h])
                    for ix, [min, max] in enumerate(rsp_proc.boxes):
                        min_int = np.array(min * dims, dtype=np.int16)
                        max_int = np.array(max * dims, dtype=np.int16)

                        write_text(
                            user_view, f"{rsp_proc.labels[ix]}", min_int, fg=colors[ix]
                        )
                    victor = get_victor(rsp_proc.labels)
                    if victor >= 0:
                        cv.rectangle(
                            user_view, (0, 0), (w, h), colors[victor], thickness=25
                        )
                cv.imshow("main", user_view)

            waitKey = cv.waitKey(10)
            if waitKey > 0:
                # print(waitKey, chr(waitKey))
                if waitKey in [ord(str(i)) for i in range(10)]:
                    print(f"{waitKey=}, {key}")
                    key = int(chr(waitKey))
                if waitKey == ord("q") or waitKey == 27:
                    run = False

    except KeyboardInterrupt as e:
        print("quitting")
