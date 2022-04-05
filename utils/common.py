import json
import cv2 as cv


def get_configs(file_name="configs.json"):
    with open(file_name, "r") as f:
        return json.load(f)


def write_text(frame, org, text, fg=(200, 200, 200), bg=(80, 0, 80)):
    x, y = org
    cv.putText(
        frame,
        text,
        (x, y),
        cv.FONT_HERSHEY_PLAIN,
        fontScale=3,
        color=bg,
        thickness=6,
    )
    cv.putText(
        frame,
        text,
        (x, y),
        cv.FONT_HERSHEY_PLAIN,
        fontScale=3,
        color=fg,
        thickness=2,
    )
