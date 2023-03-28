import json
from pathlib import Path
from typing import Dict, Iterable, List, Union
import torch


def get_device() -> torch.device:
    """Get device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.has_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    return device


def read_json(path: Path) -> List[Dict[str, Union[str, int]]]:
    with open(path, "r") as f:
        data: List[Dict[str, Union[str, int]]] = json.load(f)
    return data


def save_json(path: Path, container: Iterable) -> None:
    """write dict to path."""
    print(f"Saving json to {path}")
    with open(path, "w") as outfile:
        json.dump(container, outfile, ensure_ascii=False, indent=4)


class LabelMatcher:
    """For label consistency in zero-shot classification."""

    def __init__(self, labels: List[str], task: str = "ten-dim"):
        self.labels = labels
        self.task = task

    def __call__(self, label: str, text: str) -> str:
        if self.task == "ten-dim":
            for true_label in self.labels:
                if label.lower() in true_label.lower():
                    return true_label
            print(f"Label not found: {label}, for text: {text}")
            return "neutral"

        if self.task == "hate-speech":
            if label.lower() == "off":
                return "OFF"
            print(f"Label not found: {label}, for text: {text}")
            return "NOT"

        if self.task == "sentiment":
            if label.lower() == "positive":
                return "positive"
            elif label.lower() == "negative":
                return "negative"
            print(f"Label not found: {label}, for text: {text}")
            return "neutral"

        raise ValueError(f"Task not found: {self.task}")
