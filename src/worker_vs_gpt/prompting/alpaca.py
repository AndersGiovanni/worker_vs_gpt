from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, pipeline
from langchain import PromptTemplate, LLMChain
from worker_vs_gpt.config import VICUNA_DIR
import torch
import os

def load_alpaca(temperature = 0.7, load_in_8bit = True):
    tokenizer = LlamaTokenizer.from_pretrained("chavinlo/gpt4-x-alpaca")

    base_model = LlamaForCausalLM.from_pretrained(
        "chavinlo/gpt4-x-alpaca",
        load_in_8bit=load_in_8bit,
        torch_dtype= torch.float16
        device_map='auto',
    )

    alpaca = pipeline(
        "text-generation",
        model=base_model, 
        tokenizer=tokenizer, 
        max_length=1024,
        temperature=temperature,
        top_p=0.95,
        repetition_penalty=1.2
    )
    return alpaca

def load_vicuna_13b(temperature = 0.7, load_in_8bit = False):

    tokenizer = LlamaTokenizer.from_pretrained(os.path.join(VICUNA_DIR, 'vicuna-13b'))

    base_model = LlamaForCausalLM.from_pretrained(
        os.path.join(VICUNA_DIR, 'vicuna-13b'),
        low_cpu_mem_usage=True,
        load_in_8bit=load_in_8bit,
        device_map='auto',
        torch_dtype= torch.float16
    )

    vicuna = pipeline(
        "text-generation",
        model=base_model, 
        tokenizer=tokenizer, 
        max_length=1024,
        temperature=temperature,
        top_p=0.95,
        repetition_penalty=1.2
    )
    return vicuna