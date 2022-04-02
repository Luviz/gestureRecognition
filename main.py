from sys import argv
from core import gesture_recognition, recoder
import argparse
import json


def get_configs(file_name="configs.json"):
    with open(file_name, "r") as f:
        return json.load(f)


configs = get_configs()

parser = argparse.ArgumentParser(description="Hand gestrue regcogintion software.")

parser.add_argument(
    "--ipcam",
    dest="use_ip_cam",
    action="store_true",
    help="use ip cam you can set ipCam url in ./configs.json ",
)

parser.add_argument("--recoder", action="store_true")

args = parser.parse_args()


def main():
    use_ip_cam = (args.use_ip_cam) and (configs is not None) and ("ipCam" in configs)
    cam = configs["ipCam"] if use_ip_cam else None

    if args.recoder:
        recoder(cam)
    else:
        gesture_recognition(cam)
    # print(argv)


if __name__ == "__main__":
    main()
