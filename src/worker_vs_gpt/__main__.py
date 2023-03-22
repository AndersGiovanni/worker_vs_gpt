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

import wandb
from worker_vs_gpt.config import (
    TrainerConfig,
)

load_dotenv()

cs = ConfigStore.instance()
cs.store(name="config", node=TrainerConfig)


# @click.command()
# @click.version_option()
@hydra.main(version_base=None, config_path="conf", config_name="config_trainer.yaml")
def main(cfg: TrainerConfig) -> None:
    """Ten Social Dim."""

    print(cfg)

    a = 1


if __name__ == "__main__":
    # main(prog_name="ten_social_dim")  # pragma: no cover
    main()  # pragma: no cover
