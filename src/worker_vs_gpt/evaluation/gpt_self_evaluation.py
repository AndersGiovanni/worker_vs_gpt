"""
Similar to llama_self_evaluation.py, but for GPT-4.

"""

"""
The following file iterates over the data in the src/worker_vs_gpt/evaluation/subsets folder and evaluates all 
.json files in it. The evaluation is done by comparing the original text to the augmented text and the original label

All results are saved to subset results in the src/worker_vs_gpt/evaluation/subset_results folder.
"""
import json
import os
from pathlib import Path
from typing import Dict
from typing import List

from tqdm import tqdm

from worker_vs_gpt.evaluation.gpt import GPT
from worker_vs_gpt.evaluation.textpair import TextPair
from worker_vs_gpt.utils import read_json


def evaluate_subset(data: List[Dict]) -> List[Dict[str, str]]:
    output: List[Dict[str, str]] = []
    gpt = GPT()
    for entry in tqdm(data, desc="Evaluating subset"):
        tp: TextPair = TextPair(**entry)

        response: str = gpt.generate(
            original_label=tp.original_label,
            original_text=tp.original_text,
            augmented_text=tp.augmented_text,
        )

        tmp_prompt_string: str = gpt.llm_chain.json()

        # read json from tmp_prompt_string
        json_data = json.loads(tmp_prompt_string)
        system_prompt = json_data["prompt"]["messages"][0]["prompt"]["template"]
        user_prompt = json_data["prompt"]["messages"][1]["prompt"]["template"]
        tp.prompt = {"system_prompt": system_prompt, "user_prompt": user_prompt}
        tp.prompt_response = response

        output.append(tp.__dict__)

    return output


if __name__ == "__main__":
    import os
    from pathlib import Path
    from typing import Dict
    from typing import List

    from worker_vs_gpt.utils import read_json
    from worker_vs_gpt.utils import save_json

    root_path: Path = Path("src/worker_vs_gpt/evaluation/subsets/label_to_other_label")

    folders: List[str] = [
        folder for folder in os.listdir(root_path) if not folder.startswith(".")
    ]

    # sort folders alphabetically reversed
    folders.sort(reverse=True)

    print("Folders to evaluate:", folders)

    for folder_name in folders:
        folder_path: Path = root_path / folder_name

        files_to_evaluate: List[Path] = [
            file_path
            for file_path in folder_path.iterdir()
            if file_path.suffix == ".json"
        ]

        for i, file_path in enumerate(files_to_evaluate):
            print(f"Evaluating {folder_name} ({i+1}/{len(files_to_evaluate)})...")

            outpath: Path = Path(
                f"src/worker_vs_gpt/evaluation/subset_results/label_to_other_label/{folder_name}-subset/gpt/{file_path.stem}.json"
            )

            if os.path.exists(outpath):
                print(f"File already exists: {outpath}")
                continue

            data = read_json(file_path)

            evaluated_data: List[Dict[str, str]] = evaluate_subset(data)

            save_json(container=evaluated_data, path=outpath)
