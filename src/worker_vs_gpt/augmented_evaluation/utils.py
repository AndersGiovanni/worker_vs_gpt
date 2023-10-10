import json
import os
from typing import Dict


def load_json(filename: str, verbose: bool = True) -> Dict:
    if verbose:
        print(f"Loading {filename}")
    with open(filename, "r") as infile:
        data = json.load(infile)
    return data


def save_json(data: Dict, filename: str, verbose: bool = True) -> None:
    if verbose:
        print(f"Saving {filename}")
    with open(filename, "w") as outfile:
        json.dump(data, outfile, indent=4)


def create_path(path: str) -> None:
    """
    Creates a path if it doesn't exist.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def assert_path(path: str, build_path_on_break: bool = True) -> None:
    """
    Asserts that a path exists.
    """
    if build_path_on_break:
        create_path(path)

    assert os.path.exists(path), f"Path {path} does not exist."
