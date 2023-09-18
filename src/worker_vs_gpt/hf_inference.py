import json
from langchain import HuggingFaceTextGenInference, HuggingFaceHub, LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import requests
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

MODEL = "meta-llama/Llama-2-70b-chat-hf"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL}"
headers = {"Authorization": "Bearer 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"}


load_dotenv()


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


output = query(
    {
        "inputs": """<s>[INST] <<SYS>>
You are an AI assistant. Your job is to answer questions about Anders, and only about Anders. He is a PhD student at IT University of Copenhagen. Answer like you're a pirate.
<</SYS>>

Who is Anders? Please explain in great detail: [/INST]""",
        "parameters": {
            "options": {"wait_for_model": True},
            "temperature": 0.7,
            "min_length": 1000,
            "max_length": 4000,
            "top_k": 9,
            # "top_p": 0.7,
            "do_sample": True,
        },
    }
)

system_message = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=[],
        template="""
        You are the assistant of Anders Giovanni Møller, a PhD student at ITU. He is in the NERDS group. He is 28 years old. Answer very brief and like you're Anders.
        He was born in Denmark, in the city of Ringsted. He got his BSc and MSc in data science at IT University of Copenhagen. He plays handball. You must only answer questions about Anders. 
        """,
    )
)

human_message = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["text"],
        template="""Question: {text}


Answer: """,
    )
)
prompt_2 = ChatPromptTemplate.from_messages([system_message, human_message])

llm = HuggingFaceHub(
    repo_id=MODEL,
    task="text-generation",
    model_kwargs={
        "options": {"wait_for_model": True},
        "temperature": 1.0,
        "do_sample": True,
    },
)

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


chain = LLMChain(
    prompt=prompt_2,
    llm=llm,
    verbose=True,
)


print(chain)

print(
    chain.run(
        {
            "text": "How old is Anders and where was he born?",
        },
        callbacks=[StreamingStdOutCallbackHandler()],
    ),
    chain.run(
        {
            "text": "What education does he have?",
        },
        callbacks=[StreamingStdOutCallbackHandler()],
    ),
)
