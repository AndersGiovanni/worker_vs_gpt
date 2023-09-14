from typing import List

import datasets
import torch
from transformers import AutoTokenizer

from worker_vs_gpt.data_processing.dataclass import DataClassWorkerVsGPT


torch.manual_seed(42)

from worker_vs_gpt.config import DATA_DIR


class Socket_Dataset(DataClassWorkerVsGPT):
    """Dataclass for SOCKET datasets (they're all formatted the same way)."""

    def __init__(
        self,
        task: str,
        filename: str,
        is_augmented: bool = False,
    ) -> None:
        super().__init__(DATA_DIR / f"{task}/{filename}", is_augmented)
        self.task = task
        self.filename = filename
        self.labels = self._get_labels()
        self.is_augmented = is_augmented

    def preprocess(self, model_name: str) -> None:
        # Convert labels to list of ints
        self.data = self.data.map(
            lambda x: {"labels": self._label_preprocessing(x["target"])},
        )

        # Define tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # tokenize the text
        if self.is_augmented:
            text_column = "augmented_text"
        else:
            text_column = "text"

        self.data = self.data.map(
            lambda x: tokenizer(x[text_column], truncation=True, padding=True),
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

    def _get_labels(self) -> List[str]:
        """Get list of labels from dataset."""
        with open(DATA_DIR / f"{self.task}/data/label_list.txt", "r") as f:
            labels = [line.strip() for line in f.readlines()]
        return labels


if __name__ == "__main__":
    print("Hello world!")

    dataset = Socket_Dataset(
        task="talkdown-pairs",
        filename="train.json",
        is_augmented=False,
    )

    print(dataset.data)
    print(dataset.labels)
    print(dataset.label_to_idx_mapper())

    a = 1
