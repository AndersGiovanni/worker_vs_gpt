"""Command-line interface."""
from typing import List
import hydra
import numpy as np
import torch
from dotenv import load_dotenv
from hydra.core.config_store import ConfigStore
import random
import copy

from worker_vs_gpt.data_processing import (
    dataclass_hate_speech,
    dataclass_sentiment,
    dataclass_ten_dim,
    dataclass_analyse_tal,
)

from worker_vs_gpt.data_processing.dataclass_socket import Socket_Dataset

from worker_vs_gpt.config import (
    HATE_SPEECH_DATA_DIR,
    SENTIMENT_DATA_DIR,
    TEN_DIM_DATA_DIR,
)

import wandb
from worker_vs_gpt.config import TrainerConfig

from worker_vs_gpt.classification.trainers import ExperimentTrainer
from datasets import concatenate_datasets

load_dotenv()

cs = ConfigStore.instance()
cs.store(name="config", node=TrainerConfig)

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


# @click.command()
# @click.version_option()
@hydra.main(version_base=None, config_path="conf", config_name="config_datasize.yaml")
def main(cfg: TrainerConfig) -> None:
    """Ten Social Dim."""

    print(cfg)

    # can be 'analyse-tal', 'hate-speech', 'sentiment', 'ten-dim'
    if cfg.dataset == "hate-speech":
        dataset = dataclass_hate_speech.HateSpeechDataset(
            HATE_SPEECH_DATA_DIR / "train.json"
        )
        test_dataset = dataclass_hate_speech.HateSpeechDataset(
            HATE_SPEECH_DATA_DIR / "test.json"
        )
        base_dataset = dataclass_hate_speech.HateSpeechDataset(
            HATE_SPEECH_DATA_DIR / "base.json"
        )
        augmented_dataset = dataclass_hate_speech.HateSpeechDataset(
            HATE_SPEECH_DATA_DIR
            / f"{cfg.sampling}_{cfg.augmentation_model}_augmented.json",
            is_augmented=True,
        )
    elif cfg.dataset == "sentiment":
        dataset = dataclass_sentiment.SentimentDataset(
            SENTIMENT_DATA_DIR / "train.json"
        )
        test_dataset = dataclass_sentiment.SentimentDataset(
            SENTIMENT_DATA_DIR / "test.json"
        )
        base_dataset = dataclass_sentiment.SentimentDataset(
            SENTIMENT_DATA_DIR / "base.json"
        )
        augmented_dataset = dataclass_sentiment.SentimentDataset(
            SENTIMENT_DATA_DIR
            / f"{cfg.sampling}_{cfg.augmentation_model}_augmented.json",
            is_augmented=True,
        )
    elif cfg.dataset == "ten-dim":
        dataset = dataclass_ten_dim.SocialDataset(TEN_DIM_DATA_DIR / "train.json")
        test_dataset = dataclass_ten_dim.SocialDataset(TEN_DIM_DATA_DIR / "test.json")
        base_dataset = dataclass_ten_dim.SocialDataset(TEN_DIM_DATA_DIR / "base.json")
        augmented_dataset = dataclass_ten_dim.SocialDataset(
            TEN_DIM_DATA_DIR
            / f"{cfg.sampling}_{cfg.augmentation_model}_augmented.json",
            is_augmented=True,
        )
    elif cfg.dataset in [
        "crowdflower",
        "empathy#empathy_bin",
        "hayati_politeness",
        "hypo-l",
        "questionintimacy",
        "same-side-pairs",
        "talkdown-pairs",
    ]:
        dataset = Socket_Dataset(
            task=cfg.dataset,
            filename="train.json",
        )
        test_dataset = Socket_Dataset(
            task=cfg.dataset,
            filename="test.json",
        )
        base_dataset = Socket_Dataset(
            task=cfg.dataset,
            filename="base.json",
        )
        if cfg.use_augmented_data:
            raise ValueError("Augmented data not available for this dataset")
        augmented_dataset = Socket_Dataset(
            task=cfg.dataset,
            filename="train.json",  # Placeholder
            # filename=f"{cfg.sampling}_{cfg.augmentation_model}_augmented.json",
            is_augmented=False,  # Placeholder - should be True when augmented data is available
        )
    else:
        raise ValueError("Dataset not found")

    dataset.data["test"] = test_dataset.data["train"]
    dataset.data["base"] = base_dataset.data["train"]
    dataset.data["original_train"] = dataset.data["train"]
    dataset.data["augmented_train"] = augmented_dataset.data["train"]

    dataset.preprocess(model_name=cfg.ckpt)

    # Concatenate data - either original or augmented together with base
    if cfg.use_augmented_data:
        dataset.data["total_train"] = dataset.data["augmented_train"]
    else:
        dataset.data["total_train"] = dataset.data["original_train"]

    total_train_length = len(dataset.data["total_train"])

    # Specify the length of train and validation set
    validation_pct = 0.1
    validation_length = int(validation_pct * total_train_length)
    total_train_length = total_train_length - validation_length

    # generate list of indices to slice from
    indices_pct = list(np.linspace(0.0, 1.0, 10))

    # Select only indices with value 5000 or less
    shuffle_seeds: List[int] = random.sample(range(0, 100), 5)

    for idx_pct in indices_pct:
        dataset_copy = copy.deepcopy(dataset)

        for seed in shuffle_seeds:
            # Shuffle original train and augmented train
            dataset_copy.data["train"] = dataset_copy.data["total_train"].shuffle(seed)

            # Number of samples to select
            idx = int(idx_pct * total_train_length)

            dataset_copy.datasize_split(idx, validation_length)

            model = ExperimentTrainer(data=dataset_copy, config=cfg)

            model.train()

            model.test()

            wandb.finish()


if __name__ == "__main__":
    main()  # pragma: no cover
