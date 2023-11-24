import logging
import os
import time
from dataclasses import dataclass
from typing import Dict
from typing import List

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from huggingface_hub.inference._text_generation import OverloadedError
from huggingface_hub.utils._errors import HfHubHTTPError  # Import the HfHubHTTPError
from transformers import AutoTokenizer


load_dotenv(".env", verbose=True, override=True)
# add format with date and time
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

# create the log file in the current directory
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class Llama:
    ######### Using Huggingface and the Llama models #########
    logging.info("Loading Llama model...")
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
                    logger.info("Overloaded, trying again in 0.5 seconds...")
                    time.sleep(0.5)
                    continue
                else:
                    raise OverloadedError
            except HfHubHTTPError as e:
                if e.response.status_code == 502:
                    logger.error(
                        "502 Server Error: Bad Gateway. Trying again in 1 second..."
                    )
                    time.sleep(1)
                    continue
                else:
                    raise e


# Ensure you have proper logging configuration
logging.basicConfig(level=logging.INFO)


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
