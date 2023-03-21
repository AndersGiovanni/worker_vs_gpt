from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import datasets
import torch
from torch.utils.data import Dataset


class DataClassWorkerVsGPT(Dataset):
    """Dataclass abstraction for the worker_vs_gpt datasets. This gives us a unified framework."""

    def __init__(self, path: Union[Path, None]) -> None:
        super().__init__()
        self.data: datasets.Dataset = datasets.load_dataset(
            "json", data_files=str(path)
        )
        self.labels: Union[None, List[str]] = None

    def preprocess(self) -> None:
        """This function should be overwritten by the child class.
        It should preprocess the data for the model.
        This includes tokenization, label preprocessing, and column formatting"""
        raise NotImplementedError

    def get_data(self) -> datasets.Dataset:
        return self.data

    def label_to_idx_mapper(self) -> Dict[str, int]:
        assert self.labels is not None, "Labels are not set"

        return {label: idx for label, idx in enumerate(self.labels)}

    def idx_to_label_mapper(self) -> Dict[int, str]:
        assert self.labels is not None, "Labels are not set"

        return {idx: label for label, idx in enumerate(self.labels)}

    def set_labels(self, labels: List[str]) -> None:
        self.labels = labels

    def train_test_split(
        self,
        train_size: float = 0.9,
        seed: int = 42,
        test_split_name: str = "validation",
    ) -> None:
        """Split the dataset into train, validation, and test
        Parameters
        ----------
        train_size : float, optional
            Size of the train set, by default 0.9
        seed : int, optional
            Seed for the random split, by default 42
        Returns
        -------
        None
        """
        ds_splitter = self.data["train"].train_test_split(
            test_size=1 - train_size, seed=seed
        )
        train = ds_splitter["train"]
        test = ds_splitter["test"]

        self.data["train"] = train
        self.data[test_split_name] = test
