"""Command-line interface."""
from typing import List
import hydra
import numpy as np
import torch
from dotenv import load_dotenv
from hydra.core.config_store import ConfigStore
import random
from datasets import concatenate_datasets

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
from worker_vs_gpt.config import SetfitParams

from worker_vs_gpt.classification.setfit_trainer import SetFitClassificationTrainer

load_dotenv()

cs = ConfigStore.instance()
cs.store(name="config", node=SetfitParams)

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


# @click.command()
# @click.version_option()
@hydra.main(version_base=None, config_path="conf", config_name="config_setfit.yaml")
def main(cfg: SetfitParams) -> None:
    """Ten Social Dim. SetFit Classification."""

    print(cfg)

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
        labels: List[str] = ["NOT", "OFF"]
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
        labels: List[str] = [
            "negative",
            "neutral",
            "positive",
        ]
    elif cfg.dataset == "ten-dim":
        dataset = dataclass_ten_dim.SocialDataset(TEN_DIM_DATA_DIR / "train.json")
        test_dataset = dataclass_ten_dim.SocialDataset(TEN_DIM_DATA_DIR / "test.json")
        base_dataset = dataclass_ten_dim.SocialDataset(TEN_DIM_DATA_DIR / "base.json")
        augmented_dataset = dataclass_ten_dim.SocialDataset(
            TEN_DIM_DATA_DIR
            / f"{cfg.sampling}_{cfg.augmentation_model}_augmented.json",
            is_augmented=True,
        )
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
        ]
    else:
        raise ValueError("Dataset not found")

    dataset.data["test"] = test_dataset.data["train"]
    dataset.data["base"] = base_dataset.data["train"]
    dataset.data["original_train"] = dataset.data["train"]
    dataset.data["augmented_train"] = augmented_dataset.data["train"]

    dataset.setfit_preprocess(model_name=cfg.ckpt)

    # We have to map the augmented data to the same name as the original data (SetFit only works with one name)
    if cfg.dataset == "ten-dim":
        dataset.data["augmented_train"] = dataset.data["augmented_train"].map(
            lambda x: {"h_text": x["augmented_h_text"]}
        )
        cfg.text_selection = "h_text"
    elif cfg.dataset == "sentiment":
        dataset.data["augmented_train"] = dataset.data["augmented_train"].map(
            lambda x: {"text": x["augmented_text"]}
        )
        cfg.text_selection = "text"
    elif cfg.dataset == "hate-speech":
        dataset.data["augmented_train"] = dataset.data["augmented_train"].map(
            lambda x: {"tweet": x["augmented_tweet"]}
        )
        cfg.text_selection = "tweet"
    else:
        raise ValueError("Dataset not found")

    # Specify the length of train and validation set
    validation_length = 750

    dataset.prepare_dataset_setfit(
        f"{cfg.experiment_type}", validation_length=validation_length
    )

    if cfg.experiment_type == "crowdsourced":
        name = "crowdsourced"
    else:
        name = f"{cfg.augmentation_model}_{cfg.sampling}_{cfg.experiment_type}"

    wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=name,
        group=f"{cfg.dataset}",
        config={**cfg},
        tags=[cfg.experiment_type, cfg.sampling, cfg.augmentation_model],
    )

    model = SetFitClassificationTrainer(dataset=dataset, config=cfg, labels=labels)

    model.train(evaluate_test_set=False)

    model.test()

    wandb.finish()


if __name__ == "__main__":
    main()  # pragma: no cover
