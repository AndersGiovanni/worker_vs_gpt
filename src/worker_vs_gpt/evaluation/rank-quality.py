import json
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field
from itertools import combinations
from pathlib import Path
from typing import Dict
from typing import List

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from huggingface_hub.inference._text_generation import OverloadedError
from huggingface_hub.utils._errors import HfHubHTTPError  # Import the HfHubHTTPError
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import HumanMessagePromptTemplate
from langchain.prompts import PromptTemplate
from langchain.prompts import SystemMessagePromptTemplate
from tqdm import tqdm
from transformers import AutoTokenizer

from worker_vs_gpt.utils import read_json


CHAT_CONTEXT = """
    You are an advanced classifying AI. You are going to two texts written by a user.
    Bots texts expresses one of the following labels: knowledge, power, respect, trust, social_support, similarity_identity, fun, conflict, neutral.
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

    Both the texts contain the same label, but the label can be expressed in many ways and contain completely different words.
    Your task is to classify which of the two texts most clearly expresses the label. You have to pick one of the two texts. Draw is not an option.
    
    When providing your answer, please answer clearly with "Text 1" or "Text 2" as the first response and then provide a short description of why you respond the way you do.
    Examples of good answers are: "Text 1 - [reason]", "Text 2 - [reason]".
    """


load_dotenv(".env", verbose=True, override=True)


@dataclass
class TextPairQuality:
    """The container for the textpair quality data"""

    label: str
    src_text: str
    augmented_text: str
    score: float = 0.0
    won_outpus: List[str] = field(default_factory=list)
    bad_outputs: List[str] = field(default_factory=list)


@dataclass
class Llama:
    ######### Using Huggingface and the Llama models #########
    huggingface_model_name: str = "meta-llama/Llama-2-70b-chat-hf"
    llm = InferenceClient(
        model=huggingface_model_name,
        token=os.environ["HUGGINGFACEHUB_API_TOKEN"],  # Load with .env
    )

    tokenizer = AutoTokenizer.from_pretrained(huggingface_model_name)

    def __post_init__(self):
        # Fixing the annoying "Using sep_token, but it is not set yet."
        self.tokenizer.verbose = False
        self.tokenizer.use_default_system_prompt = False

    def generate(
        self, chat: List[Dict[str, str]], try_again_on_overload: bool = True
    ) -> str:
        prompt: str = self.tokenizer.apply_chat_template(chat, tokenize=False)

        while True:
            try:
                output: str = self.llm.text_generation(
                    prompt=prompt,
                    max_new_tokens=2048,
                    temperature=0.7,
                    repetition_penalty=1.2,
                )
                return output
            except OverloadedError:
                if try_again_on_overload:
                    time.sleep(0.5)
                    continue
                else:
                    raise OverloadedError
            except HfHubHTTPError as e:
                if e.response.status_code == 502:
                    time.sleep(1)
                    continue
                else:
                    raise e


@dataclass
class GPT:
    gpt4_llm: ChatOpenAI = field(
        default_factory=lambda: ChatOpenAI(model="gpt-4", temperature=0)
    )

    def __post_init__(self):
        self.llm_chain: LLMChain = LLMChain(
            prompt=self.get_chat_prompt(), llm=self.gpt4_llm, verbose=False
        )

    def generate(self, label: str, text1: str, text2: str) -> str:
        output: str = self.llm_chain.run(
            {
                "label": label,
                "text1": text1,
                "text2": text2,
            }
        )
        return output

    def get_chat_prompt(self) -> ChatPromptTemplate:
        system_message = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=[],
                template=CHAT_CONTEXT,
            )
        )

        human_message = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=["label", "text1", "text2"],
                template="""
            The texts express the label: {label}.
            Text 1: {text1}.
            Text 2: {text2}.
            Which text do you think most clearly expresses the label? Please answer clearly with "Text 1" or "Text 2" as the first response and then provide a short description of why you respond the way you do.
            Answer:
            """,
            )
        )
        return ChatPromptTemplate.from_messages([system_message, human_message])


def create_dataset(N: int = 2) -> None:
    """
    Takes the dataset from:
    - "data/ten-dim/balanced_gpt-4_augmented.json"
    - "data/ten-dim/balanced_llama-2-70b_augmented.json",
    samle N textpairs for each label, and write them to:
    - src/worker_vs_gpt/evaluation/rank-quality/data/gpt/
    - src/worker_vs_gpt/evaluation/rank-quality/data/llama/

    This is done to make the dataset static, so that we can compare the results.
    """

    paths: List[Path] = [
        Path("data/ten-dim/balanced_gpt-4_augmented.json"),
        Path("data/ten-dim/balanced_llama-2-70b_augmented.json"),
    ]

    for path in paths:
        data = read_json(path)
        print(f"Read {len(data)} textpairs from {path}")

        # Divide each textpair into its own dictionary of augmented labels
        augmented_labels: Dict[str, List[TextPairQuality]] = defaultdict(list)
        for _textpair in data:
            tp = TextPairQuality(
                label=_textpair["target"],
                src_text=_textpair["h_text"],
                augmented_text=_textpair["augmented_h_text"],
            )

            augmented_labels[tp.label].append(tp)

        # Sample N textpairs from each dictionary in augmented_labels
        for label, textpairs in augmented_labels.items():
            random.shuffle(textpairs)
            subset_tp: List[Dict] = [textpair.__dict__ for textpair in textpairs[:N]]

            if path.name == "balanced_gpt-4_augmented.json":
                outpath: Path = Path(
                    f"src/worker_vs_gpt/evaluation/rank-quality/data/gpt/{label}.json"
                )
            else:
                outpath: Path = Path(
                    f"src/worker_vs_gpt/evaluation/rank-quality/data/llama/{label}.json"
                )

            with open(outpath, "w") as f:
                json.dump(subset_tp, f, indent=2)
                print(f"Wrote {len(subset_tp)} textpairs to {outpath}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--create-dataset",
        help="Create a dataset of textpairs for each label",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--model",
        help="The model to asses quality for",
        choices=["gpt", "llama"],
        default="gpt",
        required=False,
    )

    args = parser.parse_args()

    if args.create_dataset:
        print("Datasets have already been created. Exiting...")
        # create_dataset(N=15)
        exit()

    datasets: List[Path] = [
        Path("src/worker_vs_gpt/evaluation/rank-quality/data/gpt"),
        Path("src/worker_vs_gpt/evaluation/rank-quality/data/llama"),
    ]

    if args.model == "gpt":
        gpt = GPT()

    if args.model == "llama":
        llama = Llama()

    for dataset in datasets:
        for path in dataset.glob("*.json"):
            # save the updated data to the same filename but in the "results" folder
            dataset_name = dataset.name
            outpath: Path = Path(
                f"src/worker_vs_gpt/evaluation/rank-quality/results/{dataset_name}/{args.model}/{path.name}"
            )

            if outpath.exists():
                print(f"Skipping {outpath} since it already exists")
                continue

            data = read_json(path)

            # convert the data to a list of TextPairQuality objects
            data = [
                TextPairQuality(
                    label=textpair["label"],
                    src_text=textpair["src_text"],
                    augmented_text=textpair["augmented_text"],
                    score=textpair["score"],
                )
                for textpair in data
            ]

            # all textpairs need to be matched with each other
            combs = combinations(data, 2)

            for textpair1, textpair2 in tqdm(combs, desc=f"Assessing {path.name}"):
                if args.model == "gpt":
                    assert textpair1.label == textpair2.label
                    output = gpt.generate(
                        label=textpair1.label,
                        text1=textpair1.augmented_text,
                        text2=textpair2.augmented_text,
                    )

                if args.model == "llama":
                    assert textpair1.label == textpair2.label
                    prompt = [
                        {
                            "role": "system",
                            "content": CHAT_CONTEXT,
                        },
                        {
                            "role": "user",
                            "content": f"""
                                The texts express the label: {textpair1.label}.
                                Text 1: {textpair1.augmented_text}.
                                Text 2: {textpair2.augmented_text}.
                                Which text do you think most clearly expresses the label? Please answer clearly with "Text 1" or "Text 2" as the first response and then provide a short description of why you respond the way you do.
                                Answer:
                                """,
                        },
                    ]

                    output = llama.generate(chat=prompt)

                output = output.strip()

                if output.startswith("Text 1"):
                    textpair1.score += 1
                    textpair1.won_outpus.append(output)
                elif output.startswith("Text 2"):
                    textpair2.score += 1
                    textpair2.won_outpus.append(output)
                else:
                    textpair1.bad_outputs.append(output)
                    textpair2.bad_outputs.append(output)

                    print(f"Encountered bad output: {output}")

            # convert the data back to a list of dictionaries
            data = [textpair.__dict__ for textpair in data]

            with open(outpath, "w") as f:
                json.dump(data, f, indent=2)
                print(f"Wrote {len(data)} textpairs to {outpath}")
