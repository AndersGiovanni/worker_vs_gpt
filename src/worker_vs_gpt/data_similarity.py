import json
import os
import time
from typing import Callable, Dict, List, Tuple
import itertools
import numpy as np
from dotenv import load_dotenv
import pandas as pd

from tqdm import tqdm

import hydra

from worker_vs_gpt.prompting.langchain_prompting import (
    DataTemplates,
)
from worker_vs_gpt.similarity.text_similarity import SentenceSimilarity
from transformers import AutoTokenizer
import evaluate

bleu = evaluate.load("bleu")

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
    idx_to_label_mapper = dataset.idx_to_label_mapper()

    base_dataset = dataset.data["base"].to_pandas()
    train_dataset = dataset.data["original_train"].to_pandas()
    augmented_dataset = dataset.data["augmented_train"].to_pandas()

    del dataset

    Sentence_sim = SentenceSimilarity(cfg.model)

    # Specify the length of train and validation set
    validation_length = 750
    if cfg.use_augmented_data:
        o_text = text
        text = "augmented_" + text
        total_train_length = len(augmented_dataset[text])
    else:
        total_train_length = len(train_dataset[text]) - validation_length

    aug="original"
    if cfg.use_augmented_data:
        data = augmented_dataset.iloc[:total_train_length]
        aug="augmented"
    else:
        data = pd.concat(
            [
                base_dataset,
                train_dataset.iloc[
                    validation_length : total_train_length + validation_length
                ],
            ],
        ).reset_index(drop=True)


    Sentence_sim.prepare_features_labels(data[text].values)
    embs = Sentence_sim.features.cpu().detach().numpy().round(4)
    labels = data['target'].values.astype(int)
    meta = np.vstack([labels, [idx_to_label_mapper[i] for i in labels]]).T.astype(str)
    no_neutral_idx = data[data['target']!=3].index
    no_neutral_embs = embs[no_neutral_idx,:]
    no_neutral_meta = meta[no_neutral_idx,:]
    #save numpy array as tsv file
    np.savetxt(os.path.join(
            SIMILARITY_DIR, f"{cfg.dataset}_{aug}_features.tsv"
        ), embs, delimiter="\t")
    np.savetxt(os.path.join(
            SIMILARITY_DIR, f"{cfg.dataset}_{aug}_labels.tsv"
        ), meta, delimiter="\t", fmt='%s') 
    np.savetxt(os.path.join(
            SIMILARITY_DIR, f"{cfg.dataset}_{aug}_features_no_neutral.tsv"
        ), no_neutral_embs, delimiter="\t")
    np.savetxt(os.path.join(
            SIMILARITY_DIR, f"{cfg.dataset}_{aug}_labels_no_neutral.tsv"
        ), no_neutral_meta, delimiter="\t", fmt='%s') 


    data["within_sim"] = 0
    for i, group in data.groupby(o_text):
        res = Sentence_sim.get_mean_similarity(group[text].values, group[text].values)
        data.loc[group.index, "within_sim"] = res

    Sentence_sim.compute_sim_matrix(data[o_text].values, data[text].values)
    data["pairwise_sim"] = Sentence_sim.get_similarity_pairs()

    data["augment_sbleu_group"] = data.groupby(o_text)[text].apply(
        lambda x: self_bleu(x, "mean")
    )
    data["augment_sbleu"] = data.apply(
        lambda x: bleu.compute(predictions=[x[text]], references=[x[o_text]])["bleu"],
        axis=1,
    )

    data = data[
        [
            o_text,
            "target",
            text,
            "augment_sbleu_group",
            "augment_sbleu",
            "pairwise_sim",
            "within_sim",
        ]
    ]
    data.to_json(
        os.path.join(
            SIMILARITY_DIR, f"{cfg.dataset}_{cfg.augmentation}_similarity.json"
        ),
        orient="records",
    )


def self_bleu(group, method=None):
    if len(group) <= 1:
        res = group.apply(lambda x: -1)
    elif method == "mean":
        res = group.apply(
            lambda x: bleu.compute(
                predictions=[x] * ((group != x) | (group.duplicated())).sum(),
                references=[[y] for y in group[((group != x) | (group.duplicated()))]],
            )["bleu"]
        )
    else:
        res = group.apply(
            lambda x: bleu.compute(predictions=[x], references=[group[group != x]])[
                "bleu"
            ]
            if (group == x).sum() == 1
            else 1
        )
    return res


if __name__ == "__main__":
    main()
