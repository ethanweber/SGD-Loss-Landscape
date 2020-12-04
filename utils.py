import os
import json

def make_dir_for_filename(filename):
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)

def load_from_json(filename: str):
    assert filename.endswith(".json")
    with open(filename, "r") as f:
        return json.load(f)

def write_to_json(filename: str, content: dict):
    assert filename.endswith(".json")
    with open(filename, "w") as f:
        json.dump(content, f)
