"""
Should iterate over the subset_results folder and generate the neccesary results.
"""
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from worker_vs_gpt.evaluation.textpair import TextPair
from worker_vs_gpt.utils import read_json


def prompt_response(response: str) -> str:
    response = response.strip()
    if response.startswith("Yes"):
        return "Yes"
    else:
        return "No"


label_to_label_path: Path = Path(
    "src/worker_vs_gpt/evaluation/subset_results/label_to_other_label"
)
files_to_evaluate: List[Path] = [
    file_path
    for file_path in label_to_label_path.iterdir()
    if file_path.suffix == ".json"
]

metric_df: pd.DataFrame = pd.DataFrame(
    columns=["src label", "aug label", "specificity", "accuracy"]
)

for file_path in tqdm(files_to_evaluate, desc=f"Evaluating {label_to_label_path}"):
    src_label, aug_label = file_path.stem.split("_to_")
    data = read_json(file_path)

    metrics = {
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": 0,
        "true_negative": 0,
    }

    for entry in data:
        tp = TextPair(**entry)

        labels_are_equal: bool = tp.original_label == tp.augmented_label
        prompt_says_yes: bool = prompt_response(tp.prompt_response) == "Yes"

        if prompt_says_yes and labels_are_equal:
            metrics["true_positive"] += 1
        elif prompt_says_yes and not labels_are_equal:
            metrics["false_positive"] += 1
        elif not prompt_says_yes and labels_are_equal:
            metrics["false_negative"] += 1
        elif not prompt_says_yes and not labels_are_equal:
            metrics["true_negative"] += 1

    try:
        specificity = metrics["true_negative"] / (
            metrics["true_negative"] + metrics["false_positive"]
        )
    except ZeroDivisionError:
        import numpy as np

        specificity = np.nan

    # Accuracy and specificity are the same in most cases, since the only time that there exist a
    # true positive is in the case that the labels are equal, and the only time that there exist a
    # true negative is in the case that the labels are not equal.
    # On the diagonal specificity can never calculated as no true negatives exist.
    # Perhaps on the diagonal it should be accuracy, and off the diagonal it should be specificity.
    # Confusing
    accuracy = (metrics["true_positive"] + metrics["true_negative"]) / sum(
        metrics.values()
    )

    new_rows = [
        {
            "src label": src_label,
            "aug label": aug_label,
            "specificity": specificity,
            "accuracy": accuracy,
        },
    ]

    new_rows_df = pd.DataFrame(new_rows)

    metric_df = pd.concat([metric_df, new_rows_df], ignore_index=True)

labels = metric_df["src label"].values.tolist() + metric_df["aug label"].values.tolist()
labels = sorted(list(set(labels)))

# specificity
specificity_confusion_matrix = pd.DataFrame(columns=labels, index=labels)
specificity_confusion_matrix = specificity_confusion_matrix.fillna(0)

for _, row in metric_df.iterrows():
    specificity_confusion_matrix.loc[row["src label"], row["aug label"]] = row[
        "specificity"
    ]
specificity_confusion_matrix.to_csv(
    "src/worker_vs_gpt/evaluation/assets/specificity_confusion_matrix.csv"
)

# accuracy
acc_confusion_matrix = pd.DataFrame(columns=labels, index=labels)
acc_confusion_matrix = acc_confusion_matrix.fillna(0)

for _, row in metric_df.iterrows():
    acc_confusion_matrix.loc[row["src label"], row["aug label"]] = row["accuracy"]
acc_confusion_matrix.to_csv(
    "src/worker_vs_gpt/evaluation/assets/acc_confusion_matrix.csv"
)


# draw heatmap
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

sns.heatmap(
    specificity_confusion_matrix,
    annot=True,
    ax=ax[0],
    cmap="Blues",
    cbar=False,
    square=True,
    vmin=0,
    vmax=1,
)
ax[0].set_title("Specificity")
ax[0].set_xlabel("Augmented Label")
ax[0].set_ylabel("Source Label")

sns.heatmap(
    acc_confusion_matrix,
    annot=True,
    ax=ax[1],
    cmap="Blues",
    cbar=False,
    square=True,
    vmin=0,
    vmax=1,
)
ax[1].set_title("Accuracy")
ax[1].set_xlabel("Augmented Label")
ax[1].set_ylabel("Source Label")

# Rotating the x-ticks and y-ticks to be at an angle
for a in ax:
    a.set_xticklabels(a.get_xticklabels(), rotation=45, ha="right")

plt.tight_layout()

plt.savefig("src/worker_vs_gpt/evaluation/assets/label_to_other_label.png")
