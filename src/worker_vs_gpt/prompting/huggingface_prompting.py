import pandas as pd
from transformers import AutoTokenizer
from huggingface_hub import InferenceClient
from worker_vs_gpt.config import HF_HUB_MODELS
from worker_vs_gpt.prompting.datasets_config import (
    CrowdflowerConfig,
    SameSidePairsConfig,
)
import os

from worker_vs_gpt.utils import parse_output


class HuggingfaceChatTemplate:
    def __init__(self, model_name: str):
        self.model_name: str = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.use_default_system_prompt = False

    def get_template_classification(self, system_prompt: str, task: str):
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
