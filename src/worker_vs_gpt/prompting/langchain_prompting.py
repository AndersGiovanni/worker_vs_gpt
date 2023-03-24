from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI

from dotenv import load_dotenv


load_dotenv()


class DataTemplates:
    def get_ten_dim_prompt(self) -> PromptTemplate:
        return PromptTemplate(
            input_variables=["social_dimension", "text"],
            template="""Based on the following social media text conveying {social_dimension}, write 10 new similar examples in style of a social media comment, that show the same intent. Separate
                        the texts by newline.

Text: {text}

Answer:
""",
        )

    def get_sentiment_prompt(self) -> PromptTemplate:
        return PromptTemplate(
            input_variables=["sentiment", "text"],
            template="""Based on the following social media text which has a {sentiment} sentiment, write 10 new similar examples in style of a social media comment, that has the same sentiment. Separate the texts by newline.

Text: {text}

Answer:
""",
        )

    def get_hate_speech_prompt(self) -> PromptTemplate:
        return PromptTemplate(
            input_variables=["hate_speech", "text"],
            template="""
                        WORK IN PROGRESS
                    """,
        )


if __name__ == "__main__":
    ten_dim_template = DataTemplates().get_ten_dim_prompt()

    print(
        ten_dim_template.format(
            social_dimension="FAKE SOCIAL DIMENSION", text="FAKE TEXT"
        )
    )

    llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.05)

    llm_chain = LLMChain(prompt=ten_dim_template, llm=llm)

    social_dimension = "trust"
    text = "I need you to trust that I understand my own feelings."

    output = llm_chain.run({"social_dimension": social_dimension, "text": text})

    a = 1
