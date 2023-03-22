"""Command-line interface."""
import click
import hydra
import numpy as np
import pandas as pd
import torch
from datasets import concatenate_datasets
from dotenv import load_dotenv
from hydra.core.config_store import ConfigStore
from sklearn.metrics import classification_report

from worker_vs_gpt.data_processing import (
    dataclass_hate_speech,
    dataclass_sentiment,
    dataclass_ten_dim,
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

# @click.command()
# @click.version_option()
@hydra.main(version_base=None, config_path="conf", config_name="config_trainer.yaml")
def main(cfg: TrainerConfig) -> None:
    """Ten Social Dim."""

    print(cfg)

    # can be 'analyse-tal', 'hate-speech', 'sentiment', 'ten-dim'
    if cfg.dataset == "analyse-tal":
        raise NotImplementedError
    elif cfg.dataset == "hate-speech":
        dataset = dataclass_hate_speech.HateSpeechDataset(
            HATE_SPEECH_DATA_DIR / "train.json"
        )
        test_dataset = dataclass_hate_speech.HateSpeechDataset(
            HATE_SPEECH_DATA_DIR / "test.json"
        )
    elif cfg.dataset == "sentiment":
        dataset = dataclass_sentiment.SentimentDataset(
            SENTIMENT_DATA_DIR / "train.json"
        )
        test_dataset = dataclass_sentiment.SentimentDataset(
            SENTIMENT_DATA_DIR / "test.json"
        )
    elif cfg.dataset == "ten-dim":
        dataset = dataclass_ten_dim.SocialDataset(TEN_DIM_DATA_DIR / "train.json")
        test_dataset = dataclass_ten_dim.SocialDataset(TEN_DIM_DATA_DIR / "test.json")
    else:
        raise ValueError("Dataset not found")

    dataset.data["test"] = test_dataset.data["train"]

    dataset.preprocess(model_name=cfg.ckpt)

    # Specify the length of train and validation set
    baseset_length = 500
    validation_length = 500
    total_train_length = len(dataset.data["train"]) - validation_length - baseset_length

    # generate list of indices jumping by 500, and the last index is the length of the dataset
    indices = list(range(0, total_train_length, 500)) + [total_train_length]

    dataset.make_static_baseset(size=baseset_length)

    for idx in indices:
        dataset.exp_datasize_split(idx, validation_length)

        model = ExperimentTrainer(data=dataset, config=cfg)

        model.train()

        model.test()

        wandb.finish()


if __name__ == "__main__":
    # main(prog_name="ten_social_dim")  # pragma: no cover
    main()  # pragma: no cover
