from dataclasses import dataclass, field
from typing import Dict, List

from worker_vs_gpt.evaluation.models import Llama


@dataclass
class TextPair:
    dataset: str
    label: str
    original_text: str
    augmented_text: str
    augmented_comes_from_original: bool
    promt__output: str = field(init=False)
    promt__augmented_comes_from_original: bool = field(init=False)


if __name__ == "__main__":
    from pathlib import Path

    from tqdm import tqdm
    from worker_vs_gpt.evaluation.models import Llama
    from worker_vs_gpt.label_definitions import ten_dim
    from worker_vs_gpt.utils import read_json, save_json

    def generate_chat_template(
        label: str, original_text: str, augmented_text: str
    ) -> List[Dict[str, str]]:
        return [
            {
                "role": "system",
                "content": f"""
                You are an advanced classifying AI. 
                You are tasked with classifying this question: Does the text written by the user express {label}?. 
                {label} is defined as: {ten_dim[label]}. 
                An example of a text that expresses {label} is: {original_text} but the text can vary in many ways and contain completely different words. 
                You should start your respone with a clear yes/no answer. Then in the sentence after, give a short description why you respond the way you do.
                """,
            },
            {
                "role": "user",
                "content": f"{augmented_text}",
            },
        ]

    llama: Llama = Llama(huggingface_model_name="meta-llama/Llama-2-7b-chat-hf")

    DATASET = "ten-dim"

    data = read_json(Path(f"data/{DATASET}/balanced_gpt-4_augmented_full.json"))
    results: List[Dict[str, str]] = []

    for src_content in tqdm(data[:5]):
        for aug_content in data[:5]:
            TP: TextPair = TextPair(
                dataset=DATASET,
                label=src_content["target"],
                original_text=src_content["h_text"],
                augmented_text=aug_content["augmented_h_text"],
                augmented_comes_from_original=src_content["h_text"]
                == aug_content["h_text"],
            )

            chat_template: List[Dict[str, str]] = generate_chat_template(
                label=TP.label,
                original_text=TP.original_text,
                augmented_text=TP.augmented_text,
            )

            TP.promt__output: str = llama.generate(chat=chat_template)
            TP.promt__augmented_comes_from_original = (
                "yes" == TP.promt__output.strip().lower()[:3]
            )

            results.append(TP.__dict__)

    save_json(
        container=results,
        path=Path(f"src/worker_vs_gpt/evaluation/results/{DATASET}/step1.json"),
    )
