import json
import os
import time
from typing import Callable, Dict, List

from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI

from dotenv import load_dotenv
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from tqdm import tqdm
import wandb

import hydra

from worker_vs_gpt.prompting.langchain_prompting import (
    DataTemplates,
    ClassificationTemplates,
)
from worker_vs_gpt.config import (
    HATE_SPEECH_DATA_DIR,
    SENTIMENT_DATA_DIR,
    TEN_DIM_DATA_DIR,
    PromptConfig,
)

load_dotenv()


@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="config_prompt_classification.yaml",
)
def main(cfg: PromptConfig) -> None:
    classification_templates = ClassificationTemplates()

    # Load data and template
    if cfg.dataset == "analyse-tal":
        raise NotImplementedError
    elif cfg.dataset == "hate-speech":
        # read json
        text = "tweet"  # text column
        dataset = pd.read_json(os.path.join(HATE_SPEECH_DATA_DIR, "test.json"))
        classification_prompt = classification_templates.classify_hate_speech()
    elif cfg.dataset == "sentiment":
        text = "text"  # text column
        dataset = pd.read_json(os.path.join(SENTIMENT_DATA_DIR, "test.json"))
        classification_prompt = classification_templates.classify_sentiment()
    elif cfg.dataset == "ten-dim":
        text = "h_text"  # text column (can be text or h_text)
        dataset = pd.read_json(os.path.join(TEN_DIM_DATA_DIR, "test.json"))
        classification_prompt = classification_templates.classify_ten_dim()
    else:
        raise ValueError(f"Dataset not found: {cfg.dataset}")

    llm = OpenAI(model_name=cfg.model, temperature=0.05)

    llm_chain = LLMChain(prompt=classification_prompt, llm=llm)

    # Predict
    y_pred = []
    y_probs = []
    idx = 0
    for input_text in tqdm(dataset[text]):
        # Sometimes refresh the model
        if idx % 200 == 0:
            llm = OpenAI(model_name=cfg.model, temperature=0.05)
            llm_chain = LLMChain(prompt=classification_prompt, llm=llm)

        output = llm_chain.run({"text": input_text})
        pred, prob = output.split("---")
        y_pred.append(pred)
        y_probs.append(float(prob))
        idx += 1

    # Evaluate
    y_true = dataset["target"].values

    # Get all unique labels
    labels = list(set(y_true))

    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    # roc_auc = roc_auc_score(y_true, y_pred, average="macro", multi_class="ovo")
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True)

    # Initialize wandb
    wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=f"{cfg.model}",
        group=f"{cfg.dataset}",
        tags=["prompt_classification"],
        config={
            "model": cfg.model,
            "dataset": cfg.dataset,
            "text_column": text,
        },
        notes=f"Prompt: {classification_prompt.template}",
    )

    metrics = {"test/accuracy": accuracy, "test/f1": f1}

    # Log results
    wandb.log(
        metrics,
    )

    df = pd.DataFrame(report)
    df["metric"] = df.index
    table = wandb.Table(data=df)

    wandb.log(
        {
            "classification_report": table,
        }
    )

    wandb.finish()


if __name__ == "__main__":
    main()
