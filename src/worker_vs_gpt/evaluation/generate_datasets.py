"""

THE goal is to have text pairs that can simply be iterated over and the chat structure applied to them.

The folowing things should be possible to achieve with the text pairs:

    1. Original text -> Original text.Augmented text
    2. Original text -> Other original.augmented text
    3. Original text -> Other label.original text.Augmented text

The only reason to generate datasets are to have static data such that I don't need to recreate
it every time I want to run an experiment.

"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict
from typing import List

from worker_vs_gpt.evaluation.textpair import TextPair
from worker_vs_gpt.utils import read_json


# pass N from the terminal when running the script
parser = argparse.ArgumentParser()
parser.add_argument(
    "--N",
    type=int,
    default=10,
    help="Number of textpairs to sample from each dictionary",
)
args = parser.parse_args()
N: int = args.N


ORIGINAL_TO_ORIGINAL_AUGMENTED: Dict[str, List[TextPair]] = defaultdict(list)
ORIGINAL_TO_OTHER_ORIGINAL_AUGMENTED: Dict[str, List[TextPair]] = defaultdict(list)
LABEL_TO_OTHER_LABEL: Dict[str, List[TextPair]] = defaultdict(list)

path: Path = Path("data/ten-dim/balanced_llama-2-70b_augmented.json")
data: List[Dict] = read_json(path)

for outer in data:
    for inner in data:
        tp = TextPair(
            # Original parameters
            original_label=outer["target"],
            original_text=outer["h_text"],
            # Augmented parameters
            augmented_label=inner["target"],
            augmented_text=inner["augmented_h_text"],
            # Meta
            aug_from_ori=outer["h_text"] == inner["h_text"],
        )

        # Divide the textpair into the different datasets

        # 1. Original text -> Original text.Augmented text
        if tp.original_label == tp.augmented_label and tp.aug_from_ori:
            ORIGINAL_TO_ORIGINAL_AUGMENTED[tp.original_label].append(tp)

        # 2. Original text -> Other original.augmented text
        if tp.original_label == tp.augmented_label and not tp.aug_from_ori:
            ORIGINAL_TO_OTHER_ORIGINAL_AUGMENTED[tp.original_label].append(tp)

        # 3 label to other label
        if tp.original_label != tp.augmented_label:
            LABEL_TO_OTHER_LABEL[tp.original_label].append(tp)

# 1: Sample WITHIN_LABEL textpairs from each dictionary in ORIGINAL_TO_ORIGINAL_AUGMENTED
for label, textpairs in ORIGINAL_TO_ORIGINAL_AUGMENTED.items():
    print(f"\t{len(textpairs)} textpairs for label {label}")
    random.shuffle(textpairs)
    subset_tp: List[Dict] = [textpair.__dict__ for textpair in textpairs[:N]]

    with open(
        Path(
            f"src/worker_vs_gpt/evaluation/subsets/within_label_ori_to_ori/{label}.json"
        ),
        "w",
    ) as outfile:
        json.dump(subset_tp, outfile, ensure_ascii=False, indent=4)

# 2: Sample WITHIN_LABEL textpairs from each dictionary in ORIGINAL_TO_OTHER_ORIGINAL_AUGMENTED
print("2:")
for label, textpairs in ORIGINAL_TO_OTHER_ORIGINAL_AUGMENTED.items():
    print(f"\t{len(textpairs)} textpairs for label {label}")
    random.shuffle(textpairs)
    subset_tp: List[Dict] = [textpair.__dict__ for textpair in textpairs[:N]]

    with open(
        Path(
            f"src/worker_vs_gpt/evaluation/subsets/within_label_aug_not_from_ori/{label}.json"
        ),
        "w",
    ) as outfile:
        json.dump(subset_tp, outfile, ensure_ascii=False, indent=4)

# 3
print("3:")
labels: List[str] = list(LABEL_TO_OTHER_LABEL.keys())

for label, textpairs in LABEL_TO_OTHER_LABEL.items():
    # Divide each textpair into its own dictionary of augmented labels
    augmented_labels: Dict[str, List[TextPair]] = defaultdict(list)
    for textpair in textpairs:
        augmented_labels[textpair.augmented_label].append(textpair)

    # Sample N textpairs from each dictionary in augmented_labels
    for augmented_label, textpairs in augmented_labels.items():
        random.shuffle(textpairs)
        subset_tp: List[Dict] = [textpair.__dict__ for textpair in textpairs[:N]]

        with open(
            Path(
                f"src/worker_vs_gpt/evaluation/subsets/label_to_other_label/{label}_to_{augmented_label}.json"
            ),
            "w",
        ) as outfile:
            json.dump(subset_tp, outfile, ensure_ascii=False, indent=4)
