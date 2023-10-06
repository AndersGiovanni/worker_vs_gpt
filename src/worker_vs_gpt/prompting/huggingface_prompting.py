import pandas as pd
from transformers import AutoTokenizer
from huggingface_hub import InferenceClient
from worker_vs_gpt.config import HF_HUB_MODELS, SAMESIDE_DATA_DIR, CROWDFLOWER_DATA_DIR
from worker_vs_gpt.prompting.datasets_config import (
    CrowdflowerConfig,
    SameSidePairsConfig,
)
import os

from worker_vs_gpt.utils import few_shot_sampling, parse_output


class HuggingfaceChatTemplate:
    def __init__(self, model_name: str):
        self.model_name: str = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.use_default_system_prompt = False

    def get_template_classification(self, system_prompt: str, task: str) -> str:
        chat = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": """{task}\n{few_shot}\nText: {text}\nAnswer: """.format(
                    task=task,
                    few_shot="{few_shot}",
                    text="{text}",
                ),
            },
        ]

        return self.tokenizer.apply_chat_template(chat, tokenize=False)

    def get_template_augmentation(self, system_prompt: str, task: str) -> str:
        chat = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": """{task}\nText: {text}\nAnswer: """.format(
                    task=task,
                    text="{text}",
                ),
            },
        ]

        return self.tokenizer.apply_chat_template(chat, tokenize=False)


if __name__ == "__main__":
    config = CrowdflowerConfig

    llm = InferenceClient(
        model=HF_HUB_MODELS["llama-2-70b"],
        token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
    )

    template = HuggingfaceChatTemplate(
        model_name=HF_HUB_MODELS["llama-2-70b"],
    ).get_template_classification(
        system_prompt=config.classification_system_prompt,
        task=config.classification_task_prompt,
    )

    train = pd.read_json(os.path.join(CROWDFLOWER_DATA_DIR, "train.json"))
    # few_shot_samples = few_shot_sampling(df=train, n=1, per_class_sampling=True)

    output = llm.text_generation(
        template.format(
            few_shot="",
            text="@meganmansyn Hahahaha! It's not horrible, if others were singing with I'm sure it could work. I wish I could afford my own drum set",
        ),
        max_new_tokens=25,
        temperature=0.9,
        repetition_penalty=1.2,
    )

    a = 1
