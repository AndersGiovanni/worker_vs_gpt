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


class AnalyseTalDataset(DataClassWorkerVsGPT):
    """Dataclass for analyse og tal."""

    def __init__(
        self, path: Union[Path, None], labels: List[str] = ["andet", "anerkendelse"]
    ) -> None:
        super().__init__(path)
        self.labels = labels

    def preprocess(self, model_name: str) -> None:
        # Convert labels to list of ints
        self.data = self.data.map(
            lambda x: {"labels": self._label_preprocessing(x["target"])},
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

        # Cast target to ClassLabel
        self.data = self.data.cast_column(
            "target", datasets.ClassLabel(names=self.labels)
        )


if __name__ == "__main__":
    print("Hello world!")

    path = ANALYSE_TAL_DATA_DIR / "train.json"

    data = AnalyseTalDataset(path)

    data.preprocess(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
