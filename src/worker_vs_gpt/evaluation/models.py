import os
import time
from dataclasses import dataclass
from typing import Dict, List

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from huggingface_hub.inference._text_generation import OverloadedError
from transformers import AutoTokenizer

load_dotenv(".env.example", verbose=True, override=True)


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
        self.tokenizer.use_default_system_prompt = False

    def generate(
        self, chat: List[Dict[str, str]], try_again_on_overload: bool = True
    ) -> str:
        prompt: str = self.tokenizer.apply_chat_template(chat, tokenize=False)

        while True:
            try:
                output = self.llm.text_generation(
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


if __name__ == "__main__":
    llama = Llama()
    chat_output: str = llama.generate(
        chat=[
            {"role": "user", "content": "Hello, how are you?"},
            {
                "role": "assistant",
                "content": "I'm doing great. How can I help you today?",
            },
            {
                "role": "user",
                "content": "I'd like to show off how chat templating works!",
            },
        ]
    )
    print(chat_output)
