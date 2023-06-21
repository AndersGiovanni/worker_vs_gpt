import json
import os
import time
from typing import Callable, Dict, List, Tuple


from dotenv import load_dotenv
import pandas as pd

from tqdm import tqdm

import hydra
from sklearn.metrics import confusion_matrix

from worker_vs_gpt.prompting.langchain_prompting import (
    DataTemplates,
)
import torch
from transformers import AutoTokenizer, pipeline, AutoModelForSequenceClassification
from setfit import SetFitModel


from worker_vs_gpt.data_processing import dataclass_ten_dim

from worker_vs_gpt.config import TEN_DIM_DATA_DIR, MODELS_DIR, SIMILARITY_DIR


load_dotenv()


def main() -> None:
    text = "h_text"
    test_dataset = dataclass_ten_dim.SocialDataset(TEN_DIM_DATA_DIR / "test.json")
    test_dataset.preprocess(model_name="intfloat/e5-base")
    data = test_dataset.data["train"][text]
    idx_to_label_mapper = test_dataset.idx_to_label_mapper()
    label_to_idx_mapper = test_dataset.label_to_idx_mapper()
    # loop through folders in MODELS_DIR
    for folder in os.listdir(MODELS_DIR):

        if "setfit" in folder:
            continue
            model = SetFitModel.from_pretrained(MODELS_DIR / folder)
            preds = model(data)
            preds = preds.detach().cpu().numpy()
        elif "checkpoint" in folder:
            
            model = AutoModelForSequenceClassification.from_pretrained(
                MODELS_DIR / folder
            )
            tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-base')
            pipe = pipeline(task="text-classification", model=model, tokenizer=tokenizer, device=0)
            preds = pipe(data)
            preds = [int(pred["label"][-1]) for pred in preds]

        else:
            continue   

        targets = test_dataset.data["train"]["target"].tolist()
        targets_labels = [idx_to_label_mapper[target] for target in targets]
        preds_labels = [idx_to_label_mapper[pred] for pred in preds]
        # make dataframe including targets and preds and labels
        df = pd.DataFrame(
            {
                "targets": targets,
                "preds": preds,
                "targets_labels": targets_labels,
                "preds_labels": preds_labels,
            }
        )
        # save dataframe to JSON
        df.to_json(
            os.path.join(SIMILARITY_DIR, f"{folder}_preds.json"),
            orient="records",
        )


if __name__ == "__main__":
    main()
