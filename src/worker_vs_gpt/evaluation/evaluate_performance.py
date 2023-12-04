"""
Should iterate over the subset_results folder and generate the neccesary results.
"""
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from worker_vs_gpt.evaluation.textpair import TextPair
from worker_vs_gpt.utils import read_json


def prompt_response(response: str) -> str:
    """Takes a respinse and returns whether it starts with 'yes' or something else."""
    response = response.strip()
    if response.startswith("Yes"):
        return "Yes"
    else:
        return "No"


label_to_label_path: Path = Path(
    "src/worker_vs_gpt/evaluation/subset_results/label_to_other_label"
)


subset_folders_to_evaluate: List[Path] = [
    folder_path for folder_path in label_to_label_path.iterdir() if folder_path.is_dir()
]

print(
    f"The following folders will be evaluated: {[path.stem for path in subset_folders_to_evaluate]}"
)
for subset_folder_path in subset_folders_to_evaluate:
    folders_to_evaluate: List[Path] = [
        folder_path
        for folder_path in subset_folder_path.iterdir()
        if folder_path.is_dir()
    ]

    print(
        f"\t{subset_folder_path.stem} will be evaluated with the following models: {[path.stem for path in folders_to_evaluate]}"
    )

    # make the background transparent
    fig, ax = plt.subplots(1, 2, figsize=(10, 5.5))

    for ax_idx, folder_path in enumerate(folders_to_evaluate):
        files_to_evaluate: List[Path] = [
            file_path
            for file_path in folder_path.iterdir()
            if file_path.suffix == ".json"
        ]

        print(
            f"\t\t{folder_path.stem} will be evaluated with {len(files_to_evaluate)} files"
        )

        metric_df: pd.DataFrame = pd.DataFrame(
            columns=["src label", "aug label", "accuracy/specificity"]
        )

        for file_path in files_to_evaluate:
            src_label, aug_label = file_path.stem.split("_to_")
            data = read_json(file_path, verbose=False)

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

            # Accuracy and specificity are the same in most cases, since the only time that there exist a
            # true positive is in the case that the labels are equal, and the only time that there exist a
            # true negative is in the case that the labels are not equal.
            # On the diagonal specificity can never calculated as no true negatives exist.
            # Perhaps on the diagonal it should be accuracy, and off the diagonal it should be specificity.
            # Confusing
            # Actually just use ACCURACY, but specify this is actaully specificity in all cases except the diagonal.
            accuracy = (metrics["true_positive"] + metrics["true_negative"]) / sum(
                metrics.values()
            )

            new_rows = [
                {
                    "src label": src_label,
                    "aug label": aug_label,
                    "accuracy/specificity": accuracy,
                },
            ]

            new_rows_df = pd.DataFrame(new_rows)

            metric_df = pd.concat([metric_df, new_rows_df], ignore_index=True)

        labels = (
            metric_df["src label"].values.tolist()
            + metric_df["aug label"].values.tolist()
        )
        labels = sorted(list(set(labels)))

        # model metrics
        model_metrics = pd.DataFrame(columns=labels, index=labels)
        model_metrics = model_metrics.fillna(0)

        for _, row in metric_df.iterrows():
            model_metrics.loc[row["src label"], row["aug label"]] = row[
                "accuracy/specificity"
            ]

        model_type = folder_path.stem

        model_metrics.to_csv(
            f"src/worker_vs_gpt/evaluation/assets/{subset_folder_path.stem}/{model_type}.csv"
        )

        sns.heatmap(
            model_metrics,
            annot=True,
            ax=ax[ax_idx],
            cmap="Blues",
            cbar=False,
            square=True,
            vmin=0,
            vmax=1,
        )
        ax[ax_idx].set_title(f"{model_type}")  # - accuracy/specificity")
        ax[ax_idx].set_xlabel("Augmented Label")
        ax[ax_idx].set_ylabel("Source Label")

    # Rotating the x-ticks and y-ticks to be at an angle
    for a in ax:
        a.set_xticklabels(a.get_xticklabels(), rotation=45, ha="right")

    # Make the overall title
    fig.suptitle(f"{subset_folder_path.stem} - Accuracy/Specificity")

    # TODO: Make the background transparent for better embedding in the paper

    plt.tight_layout()

    # save the figure to correct subfolder of assets
    plt.savefig(
        f"src/worker_vs_gpt/evaluation/assets/{subset_folder_path.stem}/label_to_other_label.png"
    )
    plt.close()
    plt.clf()
