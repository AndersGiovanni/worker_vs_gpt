import json
import os
import time
from typing import Callable, Dict, List, Tuple


from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

from dotenv import load_dotenv
import pandas as pd

from tqdm import tqdm

import hydra

from worker_vs_gpt.prompting.langchain_prompting import (
    DataTemplates,
)
from worker_vs_gpt.config import (
    HATE_SPEECH_DATA_DIR,
    SENTIMENT_DATA_DIR,
    TEN_DIM_DATA_DIR,
    AugmentConfig,
)

from worker_vs_gpt.utils import balanced_sample_df, parse_output

load_dotenv()


@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="config_prompt_augmentation.yaml",
)
def main(cfg: AugmentConfig) -> None:
    augmentation_templates = DataTemplates()

    # Load data and template
    if cfg.dataset == "analyse-tal":
        raise NotImplementedError
    elif cfg.dataset == "hate-speech":
        # read json
        text = "tweet"  # text column
        HS_dict = {"OFF": "offensive", "NOT": "not offensive"}
        dataset = pd.read_json(os.path.join(HATE_SPEECH_DATA_DIR, "base.json"))
        augmentation_prompt = augmentation_templates.get_hate_speech_prompt()
    elif cfg.dataset == "sentiment":
        text = "text"  # text column
        dataset = pd.read_json(os.path.join(SENTIMENT_DATA_DIR, "base.json"))
        augmentation_prompt = augmentation_templates.get_sentiment_prompt()
    elif cfg.dataset == "ten-dim":
        text = "h_text"  # text column (can be text or h_text)
        dataset = pd.read_json(os.path.join(TEN_DIM_DATA_DIR, "base.json"))
        augmentation_prompt = augmentation_templates.get_ten_dim_prompt()
        label_to_description = {
            "knowledge": "Exchange of ideas or information",
            "power": "Having power over the behavior and outcomes of another",
            "status": "Conferring status, appreciation, gratitude, or admiration upon another",
            "trust": "Will of relying on the actions or judgments of another",
            "social_support": "Giving emotional or practical aid and companionship",
            "similarity_identity": "Shared interests, motivations, outlooks or Shared sense of belonging to the same community or group",
            "fun": "Experiencing leisure, laughter, and joy",
            "conflict": "Contrast or diverging views",
            "neutral": "neutral communication",
        }
    else:
        raise ValueError(f"Dataset not found: {cfg.dataset}")

    llm = ChatOpenAI(model_name=cfg.model, temperature=0)

    llm_chain = LLMChain(prompt=augmentation_prompt, llm=llm)

    if cfg.sampling == "balanced":
        dataset = balanced_sample_df(dataset, 500)

    df = pd.DataFrame(columns=[f"{text}", "target", f"augmented_{text}"])

    for idx, input_text in tqdm(dataset[text].items()):
        # Sometimes refresh the model
        if idx % 100 == 0:
            llm = ChatOpenAI(model_name=cfg.model, temperature=0)
            llm_chain = LLMChain(prompt=augmentation_prompt, llm=llm)

        if cfg.dataset == "ten-dim":
            description = label_to_description[dataset["target"][idx]]
            output = llm_chain.run(
                {
                    "text": input_text,
                    "social_dimension": dataset["target"][idx],
                    "social_dimension_description": description,
                }
            )
        elif cfg.dataset == "sentiment":
            output = llm_chain.run(
                {
                    "text": input_text,
                    "sentiment": dataset["target"][idx],
                }
            )
        elif cfg.dataset == "hate-speech":
            label = HS_dict[dataset["target"][idx]]
            output = llm_chain.run(
                {
                    "text": input_text,
                    "hate_speech": label,
                }
            )
        else:
            raise NotImplementedError

        augmented_text = parse_output(input_string=output)
        pl = pd.DataFrame(augmented_text, columns=[f"augmented_{text}"])
        pl[text] = input_text
        pl["target"] = dataset["target"][idx]
        df = df.append(
            pl,
            ignore_index=True,
        )

    df = df.sample(frac=1, random_state=rng).reset_index(drop=True)

    if cfg.dataset == "analyse-tal":
        raise NotImplementedError
    elif cfg.dataset == "hate-speech":
        df.to_json(
            HATE_SPEECH_DATA_DIR / f"{cfg.sampling}_augmented.json", orient="records"
        )
    elif cfg.dataset == "sentiment":
        df.to_json(
            SENTIMENT_DATA_DIR / f"{cfg.sampling}_augmented.json", orient="records"
        )
    elif cfg.dataset == "ten-dim":
        df.to_json(
            TEN_DIM_DATA_DIR / f"{cfg.sampling}_augmented.json", orient="records"
        )


if __name__ == "__main__":
    main()
