"""
The following file iterates over the data in the src/worker_vs_gpt/evaluation/subsets folder and evaluates all 
.json files in it. The evaluation is done by comparing the original text to the augmented text and the original label

All results are saved to subset results in the src/worker_vs_gpt/evaluation/subset_results folder.
"""
import os
from pathlib import Path
from typing import Dict
from typing import List

from tqdm import tqdm

from worker_vs_gpt.evaluation.llama import Llama
from worker_vs_gpt.evaluation.textpair import TextPair
from worker_vs_gpt.utils import read_json


def evaluate_subset(data: List[Dict]) -> List[Dict[str, str]]:
    output: List[Dict[str, str]] = []
    for entry in tqdm(data, desc="Evaluating subset"):
        tp: TextPair = TextPair(**entry)

        prompt = [
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
            You are tasked with classifying this question: Does the text written by the user express {tp.original_label}?.  
            An example of a text that expresses {tp.original_label} is: "{tp.original_text}", but the text can vary in many ways and contain completely different words. 
            You should start your respone with a clear yes/no answer. Then in the sentence after, give a short description why you respond the way you do.
            User input sentence: {tp.augmented_text}
            Answer:
            """,
            },
        ]

        response: str = llama.generate(chat=prompt, try_again_on_overload=True)

        tp.prompt = prompt
        tp.prompt_reponse = response

        output.append(tp.__dict__)

    return output


if __name__ == "__main__":
    import os
    from pathlib import Path
    from typing import Dict
    from typing import List

    from worker_vs_gpt.evaluation.llama import Llama
    from worker_vs_gpt.utils import read_json
    from worker_vs_gpt.utils import save_json

    root_path: Path = Path("src/worker_vs_gpt/evaluation/subsets")

    folders: List[str] = [
        folder for folder in os.listdir(root_path) if not folder.startswith(".")
    ]

    print("Folders to evaluate:", folders)

    for folder_name in folders:
        folder_path: Path = root_path / folder_name

        files_to_evaluate: List[Path] = [
            file_path
            for file_path in folder_path.iterdir()
            if file_path.suffix == ".json"
        ]

        for i, file_path in enumerate(files_to_evaluate):
            print(f"Evaluating {folder_name} ({i}/{len(files_to_evaluate)})...")

            data = read_json(file_path)

            llama: Llama = Llama(
                huggingface_model_name="meta-llama/Llama-2-70b-chat-hf"
            )

            evaluated_data: List[Dict[str, str]] = evaluate_subset(data)

            save_json(
                container=evaluated_data,
                path=Path(
                    f"src/worker_vs_gpt/evaluation/subset_results/{folder_name}/{file_path.stem}.json"
                ),
            )
