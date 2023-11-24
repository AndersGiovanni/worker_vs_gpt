from dataclasses import dataclass
from dataclasses import field

from dotenv import load_dotenv
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import HumanMessagePromptTemplate
from langchain.prompts import PromptTemplate
from langchain.prompts import SystemMessagePromptTemplate


load_dotenv()


@dataclass
class GPT:
    # Use default_factory to create a new instance for each GPT instance
    gpt4_llm: ChatOpenAI = field(
        default_factory=lambda: ChatOpenAI(model="gpt-4", temperature=0)
    )

    def __post_init__(self):
        self.llm_chain: LLMChain = LLMChain(
            prompt=self.get_chat_prompt(), llm=self.gpt4_llm, verbose=True
        )

    def generate(
        self,
        original_label: str,
        original_text: str,
        augmented_text: str,
        try_again_on_overload: bool = True,
    ) -> str:
        # TODO: Add a try again on overload
        output: str = self.llm_chain.run(
            {
                "original_label": original_label,
                "original_text": original_text,
                "augmented_text": augmented_text,
            }
        )
        return output

    def get_chat_prompt(self) -> ChatPromptTemplate:
        system_message = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=[],
                template="""
                You are an advanced classifying AI. You are going to receive a text written by a user.
                Each text expresses one of the following labels: knowledge, power, respect, trust, social_support, similarity_identity, fun, conflict, neutral.
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
                """,
            )
        )

        human_message = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=["original_label", "original_text", "augmented_text"],
                template="""
            You are tasked with classifying this question: Does the text written by the user express {original_label}?.  
            An example of a text that expresses {original_label} is: "{original_text}", but the text can vary in many ways and contain completely different words. 
            You should start your respone with a clear yes/no answer. Then in the sentence after, give a short description why you respond the way you do.
            User input sentence: {augmented_text}
            Answer:
            """,
            )
        )
        return ChatPromptTemplate.from_messages([system_message, human_message])


if __name__ == "__main__":
    gpt = GPT()

    output: str = gpt.generate(
        original_label="knowledge", original_text="test-1", augmented_text="test-2"
    )
    print(output)
