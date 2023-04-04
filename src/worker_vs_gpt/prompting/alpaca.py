from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, pipeline
from langchain import PromptTemplate, LLMChain

import torch


def load_alpaca():
    tokenizer = LlamaTokenizer.from_pretrained("chavinlo/gpt4-x-alpaca")

    base_model = LlamaForCausalLM.from_pretrained(
        "chavinlo/gpt4-x-alpaca",
        load_in_8bit=True,
        device_map='auto',
    )

    alpaca = pipeline(
        "text-generation",
        model=base_model, 
        tokenizer=tokenizer, 
        max_length=1024,
        temperature=0.6,
        top_p=0.95,
        repetition_penalty=1.2
    )
    return alpaca