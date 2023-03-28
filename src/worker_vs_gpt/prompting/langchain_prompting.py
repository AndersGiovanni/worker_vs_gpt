from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI

from dotenv import load_dotenv


load_dotenv()


class DataTemplates:
    """Class for storing the templates for the different generation tasks."""

    def get_ten_dim_prompt(self) -> PromptTemplate:
        return PromptTemplate(
            input_variables=[
                "social_dimension",
                "social_dimension_description",
                "text",
            ],
            template="""The following social media text conveys the social dimension {social_dimension}. {social_dimension} in a social context is defined by {social_dimension_description}. Write 10 new semantically similar examples in style of a social media comment, that show the same intent and social dimension.
 Do NOT enumerate your answer and separate your answer by
 “///“
.

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


class ClassificationTemplates:
    """Class for storing the templates for the different classification tasks."""

    def classify_hate_speech(self) -> PromptTemplate:
        return PromptTemplate(
            input_variables=["text"],
            template="""
                        The following is a comment on a social media post. Classify whether the post is offensive (OFF) or not (NOT). Your answer must be one of the two options and how certain you are on a scale from 0 to 1. Answer in the style [answer]---[probability].

Text: {text}


Answer:
                    """,
        )

    def classify_sentiment(self) -> PromptTemplate:
        return PromptTemplate(
            input_variables=["text"],
            template="""
                        Your job is to classify the sentiment of a text. 
Classify the following social media comment into either “negative”, “neutral” or “positive”. Your answer MUST be either one of the three and how certain you are on a scale from 0 to 1. Answer in the style [answer]---[probability]. THe output must be lowercased.

Text: {text}

Answer:
                    """,
        )

    def classify_ten_dim(self) -> PromptTemplate:
        """Work in progress"""
        return PromptTemplate(
            input_variables=["text"],
            template="""
                        Your job is to classify the social dimension of a text. The social dimensions are: social support, conflict, trust, neutral, fun, respect, knowledge, power, similarity and identity.


Based on the following social media text, classify the social dimension of the text. You answer MUST only be one of the social dimensions and how certain you are on a scale from 0 to 1. Answer in the style [answer]---[probability]. Your answer MUST be exactly one of "social_support", "conflict", "trust", "neutral", "fun", "respect", "knowledge", "power", "similarity_identity". The answer must be lowercased.
Text: {text}

Answer:
                    """,
        )


if __name__ == "__main__":
    ten_dim_template = DataTemplates().get_ten_dim_prompt()

    classify_ten_dim = ClassificationTemplates().classify_sentiment()

    print(
        classify_ten_dim.format(
            text="Happy 22nd Birthday to the cuddy Peyton Siva aka PEY PEY!! #FumatuBloodline #AllStar #GoLouisville"
        )
    )

    llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.05)

    llm_chain = LLMChain(prompt=classify_ten_dim, llm=llm)

    social_dimension = "trust"
    text = [
        "Happy 22nd Birthday to the cuddy Peyton Siva aka PEY PEY!! #FumatuBloodline #AllStar #GoLouisville",
        "She would rather see Beyonce with me anyway ... I would make a great step mother.",
    ]

    for i in text:
        output = llm_chain.run({"text": i})
        print(f"Input: {i}")
        print(f"Output: {output}")
        print("-------")

    a = 1
