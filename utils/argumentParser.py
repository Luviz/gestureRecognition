import argparse
from core import recoder


class ArgumentParser:
    def __init__(self):
        helps = {"recoder": "Records hands gestures for model training."}
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
            const=recoder,
            **mode_common_settings,
            help=helps["recoder"]
        )

    def parse_args(self):
        return self.parser.parse_args()
