import json
import os
import time
from typing import Callable, Dict, List, Tuple


from dotenv import load_dotenv
import pandas as pd

from tqdm import tqdm

import hydra

from worker_vs_gpt.prompting.langchain_prompting import (
    DataTemplates,
)
from worker_vs_gpt.similarity.text_similarity import SentenceSimilarity
from transformers import AutoTokenizer

from worker_vs_gpt.data_processing import (
    dataclass_hate_speech,
    dataclass_sentiment,
    dataclass_ten_dim,
    dataclass_analyse_tal,
)

from worker_vs_gpt.config import (
    HATE_SPEECH_DATA_DIR,
    SENTIMENT_DATA_DIR,
    TEN_DIM_DATA_DIR,
    ANALYSE_TAL_DATA_DIR,
    SIMILARITY_DIR,
    SimilarityConfig,
)

load_dotenv()


@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="config_similarity.yaml",
)
def main(cfg: SimilarityConfig) -> None:
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
            ANALYSE_TAL_DATA_DIR / f"{cfg.augmentation}_FAKE.json",
            is_augmented=True,
        )
    elif cfg.dataset == "hate-speech":
        text = "tweet"
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
            HATE_SPEECH_DATA_DIR / f"{cfg.augmentation}.json",
            is_augmented=True,
        )
        labels: List[str] = ["NOT", "OFF"]
    elif cfg.dataset == "sentiment":
        text = "text"
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
            SENTIMENT_DATA_DIR / f"{cfg.augmentation}.json",
            is_augmented=True,
        )
        labels: List[str] = [
            "negative",
            "neutral",
            "positive",
        ]
    elif cfg.dataset == "ten-dim":
        text = "h_text"
        dataset = dataclass_ten_dim.SocialDataset(TEN_DIM_DATA_DIR / "train.json")
        test_dataset = dataclass_ten_dim.SocialDataset(TEN_DIM_DATA_DIR / "test.json")
        base_dataset = dataclass_ten_dim.SocialDataset(TEN_DIM_DATA_DIR / "base.json")
        augmented_dataset = dataclass_ten_dim.SocialDataset(
            TEN_DIM_DATA_DIR / f"{cfg.augmentation}.json",
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

    dataset.preprocess(model_name=cfg.model)

    Sentence_sim = SentenceSimilarity(cfg.model)

    similarities = Sentence_sim.compute_similarity_individual(
        dataset.data["augmented_train"], labels, text
    )

    with open(
        os.path.join(
            SIMILARITY_DIR, f"{cfg.dataset}_{cfg.augmentation}_similarity.json"
        ),
        "w",
    ) as f:
        json.dump(similarities, f)


if __name__ == "__main__":
    main()
