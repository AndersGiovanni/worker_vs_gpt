from pathlib import Path
from dataclasses import dataclass

# Defining paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = ROOT_DIR / "data"

DATA_DIR_RAW = DATA_DIR / "raw"

DATA_DIR_PROCESSED = DATA_DIR / "processed"

SRC_DIR = ROOT_DIR / "src"

MODELS_DIR = ROOT_DIR / "models"

TEN_SOCIAL_DIM_DIR = SRC_DIR / "ten_social_dim"


# Defining dataclasses
@dataclass
class Experiment:
    """Experiment parameters."""

    exp_type: str
    text_selection: str
    label_strategy: int
    ckpt: str
    dataset_train: str
    dataset_test: str
    dataset_augmented: str
    wandb_project: str
    wandb_entity: str
    use_augmented_data: bool


@dataclass
class SetfitParams:
    """Setfit parameters."""

    batch_size: int
    lr_body: float
    lr_head: float
    num_iterations: int
    num_epochs_body: int
    num_epochs_head: int
    ckpt: str
    text_selection: str
    wandb_project: str
    wandb_entity: str


@dataclass
class MultilabelParams:
    """Multilabel parameters."""

    batch_size: int
    lr: float
    num_epochs: int
    ckpt: str
    weight_decay: float
    text_selection: str
    wandb_project: str
    wandb_entity: str


@dataclass
class TrainersConfig:
    """Trainers configuration."""

    experiment: Experiment
    setfit_params: SetfitParams
    multilabel_params: MultilabelParams
