from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from pathlib import Path
from typing import Dict
from typing import List

from tqdm import tqdm

from worker_vs_gpt.evaluation.models import Llama
from worker_vs_gpt.utils import read_json
from worker_vs_gpt.utils import save_json


llama: Llama = Llama(huggingface_model_name="meta-llama/Llama-2-70b-chat-hf")


def generate_chat_template_step_1(
    original_label: str, original_text: str, augmented_text: str
) -> List[Dict[str, str]]:
    """Generates a chat template for step 1 of the evaluation"""
    return [
        {
            "role": "system",
            "content": """
                You are an advanced classifying AI. You are going to receive a text written by a user.
                Each text expresses one of the following labels: knowledge, power, respect, trust, social_support, similarity_identity, fun, conflict, neutral.
                The following is the definitions of the labels:
                - knowledge: Exchange of ideas or information,
                - power: Having power over the behavior and outcomes of another,
                - respect: Conferring status, appreciation, gratitude, or admiration upon another,
                - trust: Will of relying on the actions or judgments of another,
                - social_support: Giving emotional or practical aid and companionship,
                - similarity_identity: Shared interests, motivations, outlooks or Shared sense of belonging to the same community or group,
                - fun: Experiencing leisure, laughter, and joy,
                - conflict: Contrast or diverging views,
                - neutral: neutral communication
                """,
        },
        {
            "role": "user",
            "content": f"""
            You are tasked with classifying this question: Does the text written by the user express {original_label}?.  
            An example of a text that expresses {original_label} is: "{original_text}", but the text can vary in many ways and contain completely different words. 
            You should start your respone with a clear yes/no answer. Then in the sentence after, give a short description why you respond the way you do.
            User input sentence: {augmented_text}
            Answer:
            """,
        },
    ]


@dataclass
class TextPair:
    dataset: str
    original_label: str
    original_text: str
    augmented_label: str
    augmented_text: str
    augmented_comes_from_original: bool
    promt__output: str = None
    promt__augmented_comes_from_original: bool = None
    timestamp: str = field(default_factory=lambda: str(datetime.now()))

    def __post_init__(self):
        # Set the promt__output
        if self.promt__output is None:
            chat_template: List[Dict[str, str]] = generate_chat_template_step_1(
                original_label=self.original_label,
                original_text=self.original_text,
                augmented_text=self.augmented_text,
            )

            self.promt__output = llama.generate(
                chat=chat_template, try_again_on_overload=True
            )

        # Set the promt__augmented_comes_from_original
        if (
            self.promt__augmented_comes_from_original is None
            and self.promt__output is not None
        ):
            self.promt__augmented_comes_from_original = (
                "yes" == self.promt__output.strip().lower()[:3]
            )


def split_data_based_on_label(
    data: List[Dict[str, str]]
) -> Dict[str, List[Dict[str, str]]]:
    """Splits the data into a dictionary with the labels as keys and the data as values"""
    results: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for d in data:
        results[d["target"]].append(d)
    return results


def run_step_1(DATASET: str = "ten-dim") -> None:
    """Runs the step 1 experiment"""

    DATASET = "ten-dim"

    data: List[Dict[str, str]] = read_json(
        Path(f"data/{DATASET}/balanced_gpt-4_augmented_full.json")
    )

    target_split: Dict[str, List[Dict[str, str]]] = split_data_based_on_label(data)

    for target_label, target_subset in target_split.items():
        results: List[Dict[str, str]] = []

        # save the data subset to a file
        save_json(
            path=Path(
                f"src/worker_vs_gpt/evaluation/data/{DATASET}/step1-{target_label}.json"
            ),
            container=target_subset,
            verbose=False,
        )

        # iterate over the data subset
        for src_content in tqdm(target_subset, desc=f"{target_label}"):
            for aug_content in target_subset:
                # Create a textpair. This will also generate the promt__output and promt__augmented_comes_from_original
                TP: TextPair = TextPair(
                    dataset=DATASET,
                    original_label=src_content["target"],
                    original_text=src_content["h_text"],
                    augmented_label=aug_content["target"],
                    augmented_text=aug_content["augmented_h_text"],
                    augmented_comes_from_original=(
                        src_content["h_text"] == aug_content["h_text"]
                    ),
                )

                results.append(TP.__dict__)

        save_json(
            container=results,
            path=Path(
                f"src/worker_vs_gpt/evaluation/results/{DATASET}/step1-{target_label}.json"
            ),
            verbose=False,
        )


if __name__ == "__main__":
    run_step_1(DATASET="ten-dim")
