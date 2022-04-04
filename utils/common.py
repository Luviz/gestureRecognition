import json


def get_configs(file_name="configs.json"):
    with open(file_name, "r") as f:
        return json.load(f)
