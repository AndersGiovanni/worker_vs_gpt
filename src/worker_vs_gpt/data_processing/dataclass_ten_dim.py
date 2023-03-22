from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import datasets
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from worker_vs_gpt.data_processing.dataclass import DataClassWorkerVsGPT

from worker_vs_gpt.config import ANALYSE_TAL_DATA_DIR
from worker_vs_gpt.config import HATE_SPEECH_DATA_DIR
from worker_vs_gpt.config import MODELS_DIR
from worker_vs_gpt.config import SENTIMENT_DATA_DIR
from worker_vs_gpt.config import TEN_DIM_DATA_DIR

torch.manual_seed(42)


@dataclass
class SocialDataset(DataClassWorkerVsGPT):
    """Ten-dim dataset class."""

    def __init__(
        self,
        path: Union[Path, None],
        labels: List[str] = [
            "social_support",
            "conflict",
            "trust",
            "neutral",
            "fun",
            "respect",
            "knowledge",
            "power",
            "similarity_identity",
        ],
    ) -> None:
        # If None we just wanna use the class to preprocess data (prompt classification)
        super().__init__(path)
        self.labels = labels

    def preprocess(
        self,
        model_name: str,
        text_selection: str = "h_text",
        use_neutral_column: bool = True,
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
        use_neutral_column : bool, optional
            Whether we want to use the 'neutral' column, by default True
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

        # if use_neutral_column is False, remove the "neutral" label
        if not use_neutral_column:
            self.labels.remove("neutral")

        # Check if valid input
        assert text_selection in ["text", "h_text"], ValueError(
            "Invalid text selection"
        )

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

        # Convert labels to ints
        self.data = self.data.map(
            lambda x: {"labels": self._label_preprocessing(x["label"])},
        )

        # Format columns to torch tensors
        self.data.set_format("torch")

        # Format labels column to torch tensor with dtype float32
        self.data = self.data.map(
            lambda x: {"float_labels": x["labels"].to(torch.float)},
            remove_columns=["labels"],
        ).rename_column("float_labels", "labels")


if __name__ == "__main__":
    print("Hello world!")

    path = TEN_DIM_DATA_DIR / "labeled_dataset_multiclass.json"

    data = SocialDataset(path)

    data.preprocess(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")

    # Make static baseset
    data.make_static_baseset()

    # Specify the length of train and validation set
    baseset_length = 500
    validation_length = 500
    total_train_length = len(data.data["train"]) - validation_length - baseset_length

    # generate list of indices jumping by 500, and the last index is the length of the dataset
    indices = list(range(0, total_train_length, 500)) + [total_train_length]

    for idx in indices:
        data.exp_datasize_split(idx, validation_length)
        print(data.data)

    processed_data = data.get_data()

    a = 1
