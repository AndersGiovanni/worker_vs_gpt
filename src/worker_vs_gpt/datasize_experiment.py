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

from worker_vs_gpt.config import (
    ANALYSE_TAL_DATA_DIR,
    HATE_SPEECH_DATA_DIR,
    SENTIMENT_DATA_DIR,
    TEN_DIM_DATA_DIR,
)

import wandb
from worker_vs_gpt.config import TrainerConfig

from worker_vs_gpt.classification.trainers import ExperimentTrainer

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
    if cfg.dataset == "analyse-tal":
        dataset = dataclass_analyse_tal.AnalyseTalDataset(
            ANALYSE_TAL_DATA_DIR / "train.json"
        )
        test_dataset = dataclass_analyse_tal.AnalyseTalDataset(
            ANALYSE_TAL_DATA_DIR / "test.json"
        )
        base_dataset = dataclass_analyse_tal.AnalyseTalDataset(
            ANALYSE_TAL_DATA_DIR / "base.json"
        )
        augmented_dataset = dataclass_analyse_tal.AnalyseTalDataset(
            ANALYSE_TAL_DATA_DIR
            / f"{cfg.sampling}_{cfg.augmentation_model}_augmented_FAKE.json",
            is_augmented=True,
        )
    elif cfg.dataset == "hate-speech":
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
    else:
        raise ValueError("Dataset not found")

    dataset.data["test"] = test_dataset.data["train"]
    dataset.data["base"] = base_dataset.data["train"]
    dataset.data["original_train"] = dataset.data["train"]
    dataset.data["augmented_train"] = augmented_dataset.data["train"]

    dataset.preprocess(model_name=cfg.ckpt)

    # Specify the length of train and validation set
    validation_length = 750
    if cfg.use_augmented_data:
        total_train_length = len(dataset.data["augmented_train"])
    else:
        total_train_length = len(dataset.data["train"]) - validation_length

    # generate list of indices to slice from
    indices = list(range(0, total_train_length, 500)) + [total_train_length]

    # Select only indices with value 5000 or less
    indices = [idx for idx in indices if idx <= 5000]

    shuffle_seeds: List[int] = random.sample(range(0, 100), 5)

    # Adding special tags to wandb
    cfg.tags = ["Check deviation", "SAME SEED"]
    shuffle_seeds = [42, 42, 42, 42, 42]

    for seed in shuffle_seeds:
        dataset_copy = copy.deepcopy(dataset)

        # Shuffle original train and augmented train
        dataset_copy.data["original_train"] = dataset_copy.data[
            "original_train"
        ].shuffle(seed)

        dataset_copy.data["augmented_train"] = dataset_copy.data[
            "augmented_train"
        ].shuffle(seed)

        for idx in indices:
            if cfg.use_augmented_data:
                dataset_copy.exp_datasize_split_aug(idx, validation_length)
            else:
                dataset_copy.exp_datasize_split(
                    idx, validation_length, cfg.use_augmented_data
                )

            model = ExperimentTrainer(data=dataset_copy, config=cfg)

            model.train()

            model.test()

            wandb.finish()


if __name__ == "__main__":
    main()  # pragma: no cover
