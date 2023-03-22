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


class SentimentDataset(DataClassWorkerVsGPT):
    """Dataclass for hatespeech dataset."""

    def __init__(
        self,
        path: Union[Path, None],
        labels: List[str] = [
            "negative",
            "neutral",
            "positive",
        ],
    ) -> None:
        super().__init__(path)
        self.labels: List[str] = labels

    def preprocess(self, model_name: str) -> None:
        # Convert labels to ints
        self.data = self.data.map(
            lambda x: {"labels": self._label_preprocessing(x["label"])},
        )

        # Define tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # tokenize the text
        self.data = self.data.map(
            lambda x: tokenizer(x["text"], truncation=True, padding=True),
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

        # Format columns to torch tensors
        self.data.set_format("torch")

        # Format labels column to torch tensor with dtype float32
        self.data = self.data.map(
            lambda x: {"float_labels": x["labels"].to(torch.float)},
            remove_columns=["labels"],
        ).rename_column("float_labels", "labels")

    def _label_preprocessing(self, label: str) -> List[int]:
        """Preprocessing the labels"""

        if label == "negative":
            return [1, 0, 0]
        elif label == "neutral":
            return [0, 1, 0]
        else:
            return [0, 0, 1]


if __name__ == "__main__":
    print("Hello world!")

    path = SENTIMENT_DATA_DIR / "train.json"

    data = SentimentDataset(path)

    data.set_labels(labels=["negative", "neutral", "positive"])

    data.preprocess(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")

    processed_data = data.get_data()

    a = 1
