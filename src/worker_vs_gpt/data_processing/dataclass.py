from pathlib import Path
from typing import Dict, List, Union

import datasets
import numpy as np
import torch
from datasets import concatenate_datasets
from torch.utils.data import Dataset


# Set random seeds
torch.manual_seed(42)
np.random.seed(42)


class DataClassWorkerVsGPT(Dataset):
    """Dataclass abstraction for the worker_vs_gpt datasets. This gives us a unified framework."""

    def __init__(self, path: Union[Path, None], is_augmented: bool = False) -> None:
        super().__init__()
        self.data: datasets.DatasetDict = datasets.load_dataset(
            "json",
            data_files=str(path),
        )
        self.data.shuffle(seed=42)
        self.labels: List[str] = []
        self.is_augmented: bool = False

    def preprocess(self) -> None:
        """This function should be overwritten by the child class.
        It should preprocess the data for the model.
        This includes tokenization, label preprocessing, and column formatting"""
        raise NotImplementedError

    def __getitem__(self, index: int):
        return self.data[index]

    def get_data(self) -> datasets.DatasetDict:
        return self.data

    def label_to_idx_mapper(self) -> Dict[str, int]:
        assert self.labels is not None, "Labels are not set"

        return {label: idx for idx, label in enumerate(self.labels)}

    def idx_to_label_mapper(self) -> Dict[int, str]:
        assert self.labels is not None, "Labels are not set"

        return {idx: label for idx, label in enumerate(self.labels)}

    def set_labels(self, labels: List[str]) -> None:
        self.labels = labels

    def train_test_split(
        self,
        train_size: Union[float, int] = 0.9,
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

        if isinstance(train_size, float):
            train_size = int(len(self.data["train"]) * train_size)
        if isinstance(train_size, int):
            assert train_size <= len(self.data["train"]), "Train size is too large"

        ds_splitter = self.data["train_original"].train_test_split(
            train_size=train_size, seed=seed
        )
        train = ds_splitter["train"]
        test = ds_splitter["test"]

        self.data["train"] = train
        self.data[test_split_name] = test

    def exp_datasize_split(
        self,
        train_size: int = 500,
        validation_size: int = 500,
        use_augmented_data: bool = False,
    ) -> None:
        """Split the dataset into train, validation, and test
        Parameters
        ----------
        size : int, optional
            Size of the train set, by default 500
        Returns
        -------
        None
        """

        # Assert that the size is not to large
        assert train_size + validation_size <= len(
            self.data["original_train"]
        ), f"Train and validation ({train_size + validation_size}) is to large (max: {len(self.data['original_train'])}))"

        # Select samples for validation
        self.data["validation"] = self.data["original_train"].select(
            range(validation_size)
        )

        # Select samples for train
        self.data["train"] = (
            self.data["augmented_train"].select(range(train_size))
            if use_augmented_data
            else self.data["original_train"].select(
                range(validation_size, train_size + validation_size)
            )
        )

        # Add static base set to train
        self.data["train"] = concatenate_datasets(
            [self.data["base"], self.data["train"]]
        )

    def exp_datasize_split_aug(
        self,
        train_size: int = 500,
        validation_size: int = 500,
    ) -> None:
        """Split the dataset into train, validation, and test
        Parameters
        ----------
        size : int, optional
            Size of the train set, by default 500
        Returns
        -------
        None
        """

        # Select samples for validation
        self.data["validation"] = self.data["original_train"].select(
            range(validation_size)
        )

        # Select samples for train
        # Add static base set to train
        self.data["train"] = concatenate_datasets(
            [self.data["base"], self.data["augmented_train"].select(range(train_size))]
        )

    def prepare_dataset_setfit(
        self, experiment_type: str = "crowdsourced", validation_length: int = 500
    ) -> None:
        """
        Prepare the dataset for setfit experiments

        Parameters
        ----------
        experiment_type : str, optional
            Type of experiment, by default 'crowdsourced'. Can be 'crowdsourced', 'aug' or 'both'
        validation_size : int, optional
            Size of the validation set, by default 500

        Returns
        -------
        None
        """

        # Select samples for validation
        self.data["validation"] = self.data["original_train"].select(
            range(validation_length)
        )

        # Slice the original train set
        self.data["original_train"] = self.data["original_train"].select(
            range(validation_length, len(self.data["original_train"]))
        )

        if experiment_type == "crowdsourced":
            self.data["train"] = concatenate_datasets(
                [self.data["base"], self.data["original_train"]]
            )

        elif experiment_type == "aug":
            self.data["train"] = concatenate_datasets(
                [self.data["base"], self.data["augmented_train"]]
            )
        elif experiment_type == "both":
            self.data["train"] = concatenate_datasets(
                [
                    self.data["base"],
                    self.data["augmented_train"],
                    self.data["original_train"],
                ]
            )

        else:
            raise ValueError(f"Experiment type {experiment_type} not recognized")

    def make_static_baseset(self, size: int = 500) -> None:
        """Make a static base set"""
        assert size <= len(
            self.data["train"]
        ), "The size of the base set is larger than the train set"

        # Shuffle the dataset
        self.data["train"] = self.data["train"].shuffle(seed=42)

        # Stratified train test split
        splitter = self.data["train"].train_test_split(
            train_size=size, seed=42, stratify_by_column="target"
        )

        self.data["base"] = splitter["train"]

        self.data["original_train"] = splitter["test"]

    def _label_preprocessing(self, label: Union[str, int]) -> List[int]:
        """Preprocessing the labels"""

        if isinstance(label, int):
            label = self.idx_to_label_mapper()[label]

        label_list: List[int] = [0] * len(self.labels)
        label_list[self.labels.index(label)] = 1
        return label_list
