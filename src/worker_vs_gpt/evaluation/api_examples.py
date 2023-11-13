"""How to use HF Inference and OpenAI APIs"""

from dotenv import load_dotenv

load_dotenv(".env.example", verbose=True, override=True)

import os

from huggingface_hub import InferenceClient
from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from transformers import AutoTokenizer

######### Using Huggingface and the Llama models #########

llm = InferenceClient(
    model="meta-llama/Llama-2-70b-chat-hf",
    token=os.environ["HUGGINGFACEHUB_API_TOKEN"],  # Load with .env
)


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

chat = [
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
    {"role": "user", "content": "I'd like to show off how chat templating works!"},
]

tokenizer.use_default_system_prompt = False
prompt = tokenizer.apply_chat_template(
    chat, tokenize=False
)  # This will make the prompt in the correct format for the model based on the tokenizer

output = llm.text_generation(
    prompt=prompt,
    max_new_tokens=2048,
    temperature=0.7,
    repetition_penalty=1.2,
)  # Check documentation for the different parameters

########## OpenAI using LangChain ##########


# First we define the prompt template (this is an example from the project)s
system_message = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=[],
        template="""
        You are an advanced classifying AI. You are tasked with classifying the whether the text expresses empathy.
        """,
    )
)

human_message = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["few_shot", "text"],
        template="""Based on the following text, classify whether the text expresses empathy or not. You answer MUST only be one of the two labels. Your answer MUST be exactly one of ['empathy', 'not empathy']. The answer must be lowercased.
{few_shot}
Text: {text}

Answer:
""",
    )
)
template = ChatPromptTemplate.from_messages([system_message, human_message])

llm = ChatOpenAI(
    model="gpt-3.5-turbo", temperature=0.7
)  # Check documentation for the different parameters (https://platform.openai.com/docs/models)

llm_chain = LLMChain(prompt=template, llm=llm, verbose=False)

output = llm_chain.run({"few_shot": "some few shot examples", "text": "some text"})
