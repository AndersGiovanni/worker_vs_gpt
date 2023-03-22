from dataclasses import dataclass
from pathlib import Path
from typing import Dict
from typing import List
from typing import Union

import datasets
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from worker_vs_gpt.data_processing.dataclass import DataClassWorkerVsGPT


torch.manual_seed(42)

from worker_vs_gpt.config import ANALYSE_TAL_DATA_DIR
from worker_vs_gpt.config import HATE_SPEECH_DATA_DIR
from worker_vs_gpt.config import MODELS_DIR
from worker_vs_gpt.config import SENTIMENT_DATA_DIR
from worker_vs_gpt.config import TEN_DIM_DATA_DIR


class HateSpeechDataset(DataClassWorkerVsGPT):
    """Dataclass for hatespeech dataset."""

    def __init__(self, path: Union[Path, None]) -> None:
        super().__init__(path)

    def preprocess(self, model_name: str) -> None:
        # Convert labels to ints
        self.data = self.data.map(
            lambda x: {"labels": self._label_preprocessing(x["subtask_a"])},
        )

        # Define tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # tokenize the text
        self.data = self.data.map(
            lambda x: tokenizer(x["tweet"], truncation=True, padding=True),
            batched=True,
        )
        # convert the text to tensor
        self.data = self.data.map(
            lambda x: {"input_ids": x["input_ids"]},
            batched=True,
        )

        # convert the attention mask to tensor
        self.data = self.data.map(
            lambda x: {"attention_mask": x["attention_mask"]},
            batched=True,
        )

    def _label_preprocessing(self, label: str) -> int:
        """Preprocessing the labels"""

        if label == "OFF":
            return 1
        else:
            return 0


if __name__ == "__main__":
    print("Hello world!")

    path = HATE_SPEECH_DATA_DIR / "train.json"

    data = HateSpeechDataset(path)

    data.preprocess(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")

    processed_data = data.get_data()

    a = 1
