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

        metric_df: pd.DataFrame = pd.DataFrame(columns=["src label", "aug label", "tn"])

        tp_dataframe: pd.DataFrame = pd.DataFrame(
            columns=["src label", "aug label", "tp"]
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

            # true negative dataframe
            new_rows = [
                {
                    "src label": src_label,
                    "aug label": aug_label,
                    "tn": metrics["true_negative"],
                },
            ]

            new_rows_df = pd.DataFrame(new_rows)
            metric_df = pd.concat([metric_df, new_rows_df], ignore_index=True)

            # true positive dataframe
            if src_label == aug_label:
                new_rows = [
                    {
                        "src label": src_label,
                        "aug label": aug_label,
                        "tp": metrics["true_positive"] / 100,
                    },
                ]

                new_rows_df = pd.DataFrame(new_rows)

                tp_dataframe = pd.concat([tp_dataframe, new_rows_df], ignore_index=True)

        # tp_dataframe.to_csv
        tp_dataframe.to_csv(
            f"src/worker_vs_gpt/evaluation/assets/{subset_folder_path.stem}/tp_{folder_path.stem}.csv",
            index=False,
        )

        labels = (
            metric_df["src label"].values.tolist()
            + metric_df["aug label"].values.tolist()
        )
        labels = sorted(list(set(labels)))

        # model metrics
        model_metrics = pd.DataFrame(columns=labels, index=labels)
        model_metrics = model_metrics.fillna(0)

        for _, row in metric_df.iterrows():
            model_metrics.loc[row["src label"], row["aug label"]] = row["tn"] / 100

        model_type = folder_path.stem

        model_metrics.to_csv(
            f"src/worker_vs_gpt/evaluation/assets/{subset_folder_path.stem}/{model_type}.csv"
        )

        # set the diagonal to empty
        for label in labels:
            model_metrics.loc[label, label] = None

        # plot the heatmap
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
        ax[ax_idx].set_title(
            f"{model_type.capitalize()}" if model_type != "gpt" else "GPT"
        )
        ax[ax_idx].set_xlabel("Augmented Label")
        ax[ax_idx].set_ylabel("Source Label")

    # Rotating the x-ticks and y-ticks to be at an angle
    for a in ax:
        a.set_xticklabels(a.get_xticklabels(), rotation=45, ha="right")

    # Make the overall title
    fig.suptitle(f"{subset_folder_path.stem.capitalize()} - True Negative")

    # TODO: Make the background transparent for better embedding in the paper

    plt.tight_layout()

    # save the figure to correct subfolder of assets
    plt.savefig(
        f"src/worker_vs_gpt/evaluation/assets/{subset_folder_path.stem}/label_to_other_label.png"
    )
    plt.close()
    plt.clf()

#### TP ####

# subfigure
fig, ax = plt.subplots(1, 2, sharey=True)
fig.subplots_adjust(wspace=0.1)

# gpt subset

gpt_gpt = pd.read_csv(
    "src/worker_vs_gpt/evaluation/assets/gpt-subset/tp_gpt.csv",
)
gpt_gpt = gpt_gpt.rename(columns={"tp": "GPT", "src label": "Label"})
gpt_gpt.drop(columns=["aug label"], inplace=True)

gpt_llama = pd.read_csv(
    "src/worker_vs_gpt/evaluation/assets/gpt-subset/tp_llama.csv",
)
gpt_llama = gpt_llama.rename(columns={"tp": "Llama", "src label": "Label"})
gpt_llama.drop(columns=["aug label"], inplace=True)

df = pd.merge(gpt_gpt, gpt_llama, on="Label")
df.set_index("Label", inplace=True)

# plot the heatmap
sns.heatmap(
    df,
    annot=True,
    cmap="Blues",
    cbar=False,
    square=True,
    vmin=0,
    vmax=1,
    ax=ax[0],
)
ax[0].set_title("GPT Subset")
ax[0].set_xlabel("Model")
ax[0].set_ylabel("Label")

# llama subset

llama_llama = pd.read_csv(
    "src/worker_vs_gpt/evaluation/assets/llama-subset/tp_llama.csv",
)
llama_llama = llama_llama.rename(columns={"tp": "Llama", "src label": "Label"})
llama_llama.drop(columns=["aug label"], inplace=True)

llama_gpt = pd.read_csv(
    "src/worker_vs_gpt/evaluation/assets/llama-subset/tp_gpt.csv",
)
llama_gpt = llama_gpt.rename(columns={"tp": "GPT", "src label": "Label"})
llama_gpt.drop(columns=["aug label"], inplace=True)

df = pd.merge(llama_llama, llama_gpt, on="Label")
df.set_index("Label", inplace=True)

# plot the heatmap
sns.heatmap(
    df,
    annot=True,
    cmap="Blues",
    cbar=False,
    square=True,
    vmin=0,
    vmax=1,
    ax=ax[1],
)
ax[1].set_title("Llama Subset")
ax[1].set_xlabel("Model")
ax[1].set_ylabel("Label")

plt.tight_layout()

plt.savefig("src/worker_vs_gpt/evaluation/assets/tp.png")

import matplotlib.gridspec as gridspec

#### TN ####
import matplotlib.pyplot as plt
import pandas as pd


gpt_subset_llama = pd.read_csv(
    "src/worker_vs_gpt/evaluation/assets/gpt-subset/llama.csv", index_col=0
)
gpt_subset_gpt = pd.read_csv(
    "src/worker_vs_gpt/evaluation/assets/gpt-subset/gpt.csv", index_col=0
)
llama_subset_gpt = pd.read_csv(
    "src/worker_vs_gpt/evaluation/assets/llama-subset/gpt.csv", index_col=0
)
llama_subset_llama = pd.read_csv(
    "src/worker_vs_gpt/evaluation/assets/llama-subset/llama.csv", index_col=0
)

# For all the dataframes, set the diagonal to empty
for df in [gpt_subset_llama, gpt_subset_gpt, llama_subset_gpt, llama_subset_llama]:
    for label in df.index:
        df.loc[label, label] = None


# Create the main figure and two main subplots
fig = plt.figure(figsize=(15, 10))
main_gs = gridspec.GridSpec(
    2, 1, figure=fig, wspace=0.3
)  # Increased wspace for more space between subplots


gpt_subset_gs = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=main_gs[0], wspace=0.3, hspace=0.3
)

llama_subset_gs = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=main_gs[1], wspace=0.3, hspace=0.3
)

# Create subplots within the main subplots
ax1 = fig.add_subplot(gpt_subset_gs[0])
ax2 = fig.add_subplot(gpt_subset_gs[1])
ax3 = fig.add_subplot(llama_subset_gs[0])
ax4 = fig.add_subplot(llama_subset_gs[1])

# Plot your data here
sns.heatmap(
    gpt_subset_llama,
    annot=True,
    cmap="Blues",
    cbar=False,
    square=True,
    vmin=0,
    vmax=1,
    ax=ax1,
    annot_kws={"size": 8},  # Smaller font size for numbers
)

sns.heatmap(
    gpt_subset_gpt,
    annot=True,
    cmap="Blues",
    cbar=False,
    square=True,
    vmin=0,
    vmax=1,
    ax=ax2,
    annot_kws={"size": 8},  # Smaller font size for numbers
)

sns.heatmap(
    llama_subset_gpt,
    annot=True,
    cmap="Blues",
    cbar=False,
    square=True,
    vmin=0,
    vmax=1,
    ax=ax4,
    annot_kws={"size": 8},  # Smaller font size for numbers
)

sns.heatmap(
    llama_subset_llama,
    annot=True,
    cmap="Blues",
    cbar=False,
    square=True,
    vmin=0,
    vmax=1,
    ax=ax3,
    annot_kws={"size": 8},  # Smaller font size for numbers
)

# Set the titles
ax1.set_title("GPT subset - Llama")
ax2.set_title("GPT subset - GPT")
ax3.set_title("Llama subset - Llama")
ax4.set_title("Llama subset - GPT")

# angle the x-ticks and small the font size
for a in [ax1, ax2, ax3, ax4]:
    a.set_xticklabels(a.get_xticklabels(), rotation=45, ha="right")
    a.tick_params(axis="both", which="major", labelsize=8)

# make the plot as tight as possible

plt.savefig("src/worker_vs_gpt/evaluation/assets/tn.png")
