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
    elif cfg.dataset == "ten-dim":
        text = "h_text"
        dataset = dataclass_ten_dim.SocialDataset(TEN_DIM_DATA_DIR / "train.json")
        test_dataset = dataclass_ten_dim.SocialDataset(TEN_DIM_DATA_DIR / "test.json")
        base_dataset = dataclass_ten_dim.SocialDataset(TEN_DIM_DATA_DIR / "base.json")
        augmented_dataset = dataclass_ten_dim.SocialDataset(
            TEN_DIM_DATA_DIR / f"{cfg.augmentation}.json",
            is_augmented=True,
        )
    else:
        raise ValueError("Dataset not found")

    dataset.data["test"] = test_dataset.data["train"]
    dataset.data["base"] = base_dataset.data["train"]
    dataset.data["original_train"] = dataset.data["train"]
    dataset.data["augmented_train"] = augmented_dataset.data["train"]

    dataset.preprocess(model_name=cfg.model)

    base_dataset = dataset.data["base"].to_pandas()
    train_dataset = dataset.data["original_train"].to_pandas()
    augmented_dataset = dataset.data["augmented_train"].to_pandas()

    del dataset

    Sentence_sim = SentenceSimilarity(cfg.model)

    # Specify the length of train and validation set
    validation_length = 750
    if cfg.use_augmented_data:
        text = "augmented_" + text
        total_train_length = len(augmented_dataset[text])
    else:
        total_train_length = len(train_dataset[text]) - validation_length

    # generate list of indices to slice from
    indices = list(range(0, total_train_length, 500)) + [total_train_length]

    # Select only indices with value 5000 or less
    indices = [4000]
    res = {k: [] for k in ["size", "similarity"]}

    for idx in indices:
        if cfg.use_augmented_data:
            if idx == 0:
                continue
            data = augmented_dataset.iloc[validation_length : idx + validation_length]
        else:
            data = pd.concat(
                [
                    base_dataset,
                    train_dataset.iloc[validation_length : idx + validation_length],
                ],
            )

        Sentence_sim.prepare_features_labels(data[text].values, data["target"].values)
        Sentence_sim.compute_TransRate()

        Sentence_sim.compute_sim_matrix(data[text].values, data[text].values)

        res["size"].append(idx)
        res["similarity"].append(Sentence_sim.get_similarity_sources_targets().mean())

    # turn dict into dataframe and save as json
    res = pd.DataFrame(res)
    res.to_json(
        os.path.join(
            SIMILARITY_DIR, f"{cfg.dataset}_{cfg.augmentation}_similarity.json"
        )
    )


if __name__ == "__main__":
    main()
