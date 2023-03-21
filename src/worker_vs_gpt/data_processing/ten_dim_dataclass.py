from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import datasets
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from worker_vs_gpt.data_processing.dataclass import DataClassWorkerVsGPT

torch.manual_seed(42)


@dataclass
class SocialDataset(DataClassWorkerVsGPT):
    """Ten-dim dataset class."""

    def __init__(self, path: Union[Path, None]) -> None:
        # If None we just wanna use the class to preprocess data (prompt classification)
        super().__init__(path)

    def preprocess(
        self,
        model_name: str,
        label_strategy: int = 1,
        text_selection: str = "h_text",
        use_other_column: bool = False,
    ) -> None:
        """Preprocess the data for the model. This includes tokenization, label preprocessing, and column formatting
        Parameters
        ----------
        model_name : str
            Model name to use for tokenization
        label_strategy : int, optional
            How do we preprocess the labels, by default 1
        text_selection : str, optional
            Which text do we want to use, by default "h_text"
        use_other_column : bool, optional
            Whether we want to use the 'other' column, by default False
        Raises
        ------
        ValueError
            Invalid text selection
        ValueError
            Invalid label strategy
        Returns
        -------
        None
        """
        # load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Shuffle the data
        self.data = self.data.shuffle(seed=42)

        # if use_other_column is False, remove the "other" label
        if not use_other_column:
            self.labels.remove("other")

        # Check if valid input
        assert text_selection in ["text", "h_text"], ValueError(
            "Invalid text selection"
        )
        assert label_strategy in [1, 2, 3], ValueError("Invalid label strategy")

        # tokenize the text
        self.data = self.data.map(
            lambda x: tokenizer(x[text_selection], truncation=True, padding=True),
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

        # combine all the labels into one tensor
        self.data = self.data.map(
            lambda x: {
                "labels": torch.stack([torch.tensor(x[label]) for label in self.labels])
            },
        )

        # Label preprocessing strategies
        if label_strategy == 1:
            self.data = self.data.map(
                lambda x: {"labels": self._label_strategy_1(x["labels"])},
            )
        elif label_strategy == 2:
            self.data = self.data.map(
                lambda x: {"labels": self._label_strategy_2(x["labels"])},
            )
        elif label_strategy == 3:
            self.data = self.data.map(
                lambda x: {"labels": self._label_strategy_3(x["labels"])},
            )
        else:
            raise ValueError("Invalid label strategy")

        # Format columns to torch tensors
        self.data.set_format("torch")

        # Format labels column to torch tensor with dtype float32
        self.data = self.data.map(
            lambda x: {"float_labels": x["labels"].to(torch.float)},
            remove_columns=["labels"],
        ).rename_column("float_labels", "labels")

    def _label_strategy_1(self, input: List[str]) -> List[int]:
        """If ≥ 2, set to 1. Otherwise, set to 0.
        Parameters
        ----------
        input : List[str]
            Labels for a single post
        Returns
        -------
        List[int]
            Processed labels for a single post
        """
        return [1 if int(label) >= 2 else 0 for label in input]

    def _label_strategy_2(self, input: List[str]) -> List[int]:
        """If ≥ 1, set to 1. Otherwise, set to 0.
        Parameters
        ----------
        input : List[str]
            Labels for a single post
        Returns
        -------
        List[int]
            Processed labels for a single post
        """
        return [1 if int(label) >= 1 else 0 for label in input]

    def _label_strategy_3(self, input: List[str]) -> List[int]:
        """If >= 2, set to 1, disregard all 1, otherwise set to 0.
        Parameters
        ----------
        input : List[str]
            Labels for a single post
        Returns
        -------
        List[int]
            Processed labels for a single post
        """

        return [1 if int(label) >= 2 else 0 for label in input]
