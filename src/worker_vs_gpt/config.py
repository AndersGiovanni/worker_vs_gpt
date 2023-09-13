from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Defining paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = ROOT_DIR / "data"

ANALYSE_TAL_DATA_DIR = DATA_DIR / "analyse-tal"

SENTIMENT_DATA_DIR = DATA_DIR / "sentiment"

TEN_DIM_DATA_DIR = DATA_DIR / "ten-dim"

HATE_SPEECH_DATA_DIR = DATA_DIR / "hate-speech"

CROWDFLOWER_DATA_DIR = DATA_DIR / "crowdflower"

EMPATHY_DATA_DIR = DATA_DIR / "empathy#empathy_bin"

POLITENESS_DATA_DIR = DATA_DIR / "hayati_politeness"

HYPO_DATA_DIR = DATA_DIR / "hypo-l"

INTIMACY_DATA_DIR = DATA_DIR / "questionintimacy"

SAMESIDE_DATA_DIR = DATA_DIR / "same-side-pairs"

TALKDOWN_DATA_DIR = DATA_DIR / "talkdown-pairs"

SIMILARITY_DIR = DATA_DIR / "similarity_results"

SRC_DIR = ROOT_DIR / "src"

MODELS_DIR = ROOT_DIR / "models"

WORKER_VS_GPT_DIR = SRC_DIR / "worker_vs_gpt"

LORA_WEIGHTS_DIR = MODELS_DIR / "lora"

LLAMA_CPP_DIR = MODELS_DIR / "llama_cpp"

VICUNA_DIR = MODELS_DIR / "vicuna"


@dataclass
class TrainerConfig:
    """Trainer config class."""

    ckpt: str
    dataset: str
    use_augmented_data: bool
    sampling: str
    augmentation_model: str
    wandb_project: str
    wandb_entity: str
    batch_size: int
    num_epochs: int
    lr: float
    weight_decay: float
    experiment_type: Optional[str] = None


@dataclass
class PromptConfig:
    """Config for prompting classification."""

    model: str
    dataset: str
    wandb_project: str
    wandb_entity: str


@dataclass
class AugmentConfig:
    """Config for prompting augmentation."""

    model: str
    dataset: str
    sampling: str


@dataclass
class SetfitParams:
    """Setfit parameters."""

    batch_size: int
    lr_body: float
    lr_head: float
    num_iterations: int
    num_epochs_body: int
    num_epochs_head: int
    weight_decay: float
    ckpt: str
    text_selection: str
    wandb_project: str
    wandb_entity: str
    experiment_type: str
    sampling: str
    augmentation_model: str
    dataset: str
