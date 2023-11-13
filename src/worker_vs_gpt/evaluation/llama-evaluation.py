from dotenv import load_dotenv


load_dotenv(".env.example", verbose=True, override=True)

import datetime
import os
import random

from huggingface_hub import InferenceClient
from transformers import AutoTokenizer

from worker_vs_gpt.label_definitions import ten_dim
from worker_vs_gpt.utils import assert_path
from worker_vs_gpt.utils import read_json
from worker_vs_gpt.utils import save_json


def pair_wise(dataset: str = "ten-dim") -> None:
    """Create a dataset of labels for the tendim labels and corresponding augmented labels.
    The function will have a sideeffect of creating a file in the src/worker_vs_gpt/evaluation/data/ directory.
    """

    N_to_pick = 250
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    assert_path(
        f"src/worker_vs_gpt/evaluation/data/{dataset}/", build_path_on_break=True
    )

    model = "meta-llama/Llama-2-13b-chat-hf"

    # defining models
    llm = InferenceClient(
        model=model,
        token=os.environ["HUGGINGFACEHUB_API_TOKEN"],  # Load with .env
    )

    tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer.use_default_system_prompt = False

    # reading content
    contents = read_json(f"data/{dataset}/balanced_gpt-4_augmented_full.json")

    new_content = []

    for i, content in enumerate(contents):
        print(f"{i+1}/{N_to_pick}")
        original_text: str = content["h_text"]
        label: str = content["target"]
        augmented_text: str = content["augmented_h_text"]

        for tmp_content in contents:
            tmp_original_text: str = tmp_content["h_text"]
            tmp_label: str = tmp_content["target"]
            tmp_augmented_text: str = tmp_content["augmented_h_text"]

            # Just to pick a random between 0 and 1
            n = random.random()

            if n > 0.10:
                print(f"Skipping {label} -> {tmp_label}")
                continue
            else:
                print(f"Picked {label} -> {tmp_label}")
                content["compares_to"] = tmp_label
                content["compares_to_text"] = tmp_original_text
                content["compares_to_augmented_text"] = tmp_augmented_text
                augmented_text = tmp_augmented_text
                break

        chat = [
            {
                "role": "system",
                "content": f"You are an advanced classifying AI. You are tasked with classifying the whether the text expresses {label}. {label} is defined as: {ten_dim[label]}. An example of a text that expresses {label} is: {original_text} but the text can vary in many ways and contain completely different words. You should start your respone with a clear yes/no answer, and then give a short description why respond the way you do.",
            },
            {
                "role": "user",
                "content": f"Does the following text express {label}? -> {augmented_text}",
            },
        ]

        prompt = tokenizer.apply_chat_template(
            chat, tokenize=False
        )  # This will make the prompt in the correct format for the model based on the tokenizer

        llm_response: str = llm.text_generation(
            # details=True,
            prompt=prompt,
            max_new_tokens=100,
            temperature=0.7,
            repetition_penalty=1.2,
        )  # Check documentation for the different parameters

        content["llm_response"] = llm_response
        content["model"] = model

        new_content.append(content)

        if len(new_content) == N_to_pick:
            break

    # writing content
    save_json(
        container=new_content,
        path=f"src/worker_vs_gpt/evaluation/data/{dataset}/{timestamp}-llm_responses.json",
    )


def run_model():
    # Create the chat prompts for each of the tendim labels and corresponding augmented labels
    chat = [
        {
            "role": "system",
            "content": "You are an advanced classifying AI. You are tasked with classifying the whether the text expresses empathy.",
        },
        {"role": "user", "content": "Who are you?"},
        # {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
        # {"role": "user", "content": "I'd like to show off how chat templating works!"},
    ]

    llm = InferenceClient(
        model="meta-llama/Llama-2-70b-chat-hf",
        token=os.environ["HUGGINGFACEHUB_API_TOKEN"],  # Load with .env
    )

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-chat-hf")
    tokenizer.use_default_system_prompt = False
    prompt = tokenizer.apply_chat_template(
        chat, tokenize=False
    )  # This will make the prompt in the correct format for the model based on the tokenizer

    output = llm.text_generation(
        prompt=prompt,
        max_new_tokens=50,
        temperature=0.7,
        repetition_penalty=1.2,
    )  # Check documentation for the different parameters

    print(output)


if __name__ == "__main__":
    pair_wise()
