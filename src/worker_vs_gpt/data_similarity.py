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

from worker_vs_gpt.config import (
    HATE_SPEECH_DATA_DIR,
    SENTIMENT_DATA_DIR,
    TEN_DIM_DATA_DIR,
    ANALYSE_TAL_DATA_DIR,
    DATA_DIR,
    SimilarityConfig,
)

load_dotenv()


@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="config_similarity.yaml",
)
def main(cfg: SimilarityConfig) -> None:
    # Load data
    if cfg.dataset == "analyse-tal":
        text = "tweet"  # text column
        language = "Danish"
        base_dataset = pd.read_json(os.path.join(ANALYSE_TAL_DATA_DIR, "base.json"))
        train_dataset = pd.read_json(os.path.join(ANALYSE_TAL_DATA_DIR, "train.json"))
    elif cfg.dataset == "hate-speech":
        # read json
        text = "tweet"  # text column
        augmented = "augmented_tweet"
        base_dataset = pd.read_json(os.path.join(HATE_SPEECH_DATA_DIR, "base.json"))
        train_dataset = pd.read_json(os.path.join(HATE_SPEECH_DATA_DIR, "train.json"))
        augmented_dataset = pd.read_json(
            os.path.join(SENTIMENT_DATA_DIR, f"{cfg.augmentation}.json")
        )
    elif cfg.dataset == "sentiment":
        text = "text"  # text column
        augmented = "augmented_text"
        base_dataset = pd.read_json(os.path.join(SENTIMENT_DATA_DIR, "base.json"))
        train_dataset = pd.read_json(os.path.join(SENTIMENT_DATA_DIR, "train.json"))
        augmented_dataset = pd.read_json(
            os.path.join(SENTIMENT_DATA_DIR, f"{cfg.augmentation}.json")
        )
    elif cfg.dataset == "ten-dim":
        text = "h_text"  # text column (can be text or h_text)
        augmented = "augmented_h_text"
        base_dataset = pd.read_json(os.path.join(TEN_DIM_DATA_DIR, "base.json"))
        train_dataset = pd.read_json(os.path.join(TEN_DIM_DATA_DIR, "train.json"))
        augmented_dataset = pd.read_json(
            os.path.join(SENTIMENT_DATA_DIR, f"{cfg.augmentation}.json")
        )
    else:
        raise ValueError(f"Dataset not found: {cfg.dataset}")

    Sentence_sim = SentenceSimilarity(cfg.model)

    train_size = min(len(train_dataset), 5000)
    train_dataset = train_dataset.head(train_size)

    # get all base examples
    duplicated_text = augmented_dataset[text].unique()

    # get within similarity between augmented examples generated from the same base example
    augmented_dataset["within_base_augment_sim"] = 0
    augmented_dataset["within_base_augment_sim"] = augmented_dataset.groupby(text)[
        augmented
    ].transform(lambda x: Sentence_sim.get_mean_similarity(x.values, x.values))

    # compute similarities between base examples and augmented examples
    Sentence_sim.compute_sim_matrix(
        augmented_dataset[text].values, augmented_dataset[augmented].values
    )

    augmented_dataset[
        "base_augment_pair_sim"
    ] = (
        Sentence_sim.get_similarity_pairs()
    )  # pairwise similarity between base example and augmented example
    augmented_dataset["base_augment_sim"] = Sentence_sim.get_similarity_sources_targets(
        row=False
    )  # mean similarity between augmented example and all base examples

    # remove base examples from training set
    train_oov_dataset = train_dataset[~train_dataset[text].isin(duplicated_text)]

    Sentence_sim.compute_sim_matrix(
        train_oov_dataset[text].values, augmented_dataset[augmented].values
    )
    augmented_dataset[
        "train_augment_sim"
    ] = Sentence_sim.get_similarity_sources_targets(
        row=False
    )  # mean similarity between augmented example and all non-base training examples

    Sentence_sim.compute_sim_matrix(
        augmented_dataset[augmented].values, augmented_dataset[augmented].values
    )
    augmented_dataset[
        "within_augment_sim"
    ] = (
        Sentence_sim.get_similarity_sources_targets()
    )  # within similarity between all augmented examples

    Sentence_sim.compute_sim_matrix(
        train_dataset[text].values, train_dataset[text].values
    )
    train_dataset[
        "within_train_sim"
    ] = (
        Sentence_sim.get_similarity_sources_targets()
    )  # within similarity between train examples

    # save mean and standard deviation of similarity scores in dictionary
    similarity_dict = {
        "within_base_augment_sim": {
            "mean": augmented_dataset["within_base_augment_sim"].mean(),
            "std": augmented_dataset["within_base_augment_sim"].std(),
        },
        "base_augment_pair_sim": {
            "mean": augmented_dataset["base_augment_pair_sim"].mean(),
            "std": augmented_dataset["base_augment_pair_sim"].std(),
        },
        "base_augment_sim": {
            "mean": augmented_dataset["base_augment_sim"].mean(),
            "std": augmented_dataset["base_augment_sim"].std(),
        },
        "train_augment_sim": {
            "mean": augmented_dataset["train_augment_sim"].mean(),
            "std": augmented_dataset["train_augment_sim"].std(),
        },
        "within_augment_sim": {
            "mean": augmented_dataset["within_augment_sim"].mean(),
            "std": augmented_dataset["within_augment_sim"].std(),
        },
        "within_train_sim": {
            "mean": train_dataset["within_train_sim"].mean(),
            "std": train_dataset["within_train_sim"].std(),
        },
    }
    # turn similarity dictionary into dataframe
    similarity_df = pd.DataFrame.from_dict(similarity_dict, orient="index")
    # save similarity dataframe
    similarity_df.to_csv(
        os.path.join(
            DATA_DIR,
            "similarity_results", f"{cfg.dataset}_{cfg.augmentation}_similarity.csv",
        )
    )


if __name__ == "__main__":
    main()
