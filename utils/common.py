import json
import cv2 as cv


def get_configs(file_name="configs.json"):
    with open(file_name, "r") as f:
        return json.load(f)


def write_text(
    frame,
    text,
    org,
    fg=(200, 200, 200),
    bg=(0, 0, 0),
    font=cv.FONT_HERSHEY_SIMPLEX,
    scale=1,
):
    x, y = org
    cv.putText(
        frame,
        text,
        (x, y),
        font,
        fontScale=scale,
        color=bg,
        thickness=6,
    )
    cv.putText(
        frame,
        text,
        (x, y),
        font,
        fontScale=scale,
        color=fg,
        thickness=2,
    )
