from utils import ArgumentParser, get_configs


parser = ArgumentParser()

configs = get_configs()
args = parser.parse_args()


def start_gesture_recognition(cam):
    from core.gestureRecognition import gesture_recognition

    gesture_recognition(cam)


def main():
    cam = try_get_cam_config() if args.use_ip_cam else None

    if args.callback is not None:
        args.callback(cam)
    else:
        start_gesture_recognition(cam)


def try_get_cam_config():
    return configs["ipCam"] if (configs is not None) and ("ipCam" in configs) else None


if __name__ == "__main__":
    main()
