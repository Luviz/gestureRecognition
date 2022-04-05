import argparse
from core.recoder import gesture_recoder


def start_train_gesture(*args, **kwargs):
    from core.trainModel import train_gesture

    train_gesture()


class ArgumentParser:
    def __init__(self):
        helps = {
            "recoder": "Records hands gestures for model training.",
            "train_gesture": "Starting the training based on ./gestures",
        }
        mode_common_settings = {"dest": "callback", "action": "store_const"}

        self.parser = argparse.ArgumentParser(
            description="Hand gestrue regcogintion software."
        )

        self.parser.add_argument(
            "--ipcam",
            dest="use_ip_cam",
            action="store_true",
            help="use ip cam you can set ipCam url in ./configs.json ",
        )

        parser_mode = self.parser.add_argument_group("Mode")

        parser_mode.add_argument(
            "-r",
            "--recoder",
            const=gesture_recoder,
            **mode_common_settings,
            help=helps["recoder"],
        )

        parser_mode.add_argument(
            "-t",
            "--training",
            const=start_train_gesture,
            **mode_common_settings,
            help=helps["train_gesture"],
        )

    def parse_args(self):
        return self.parser.parse_args()
