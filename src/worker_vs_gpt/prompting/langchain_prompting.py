from langchain import LLMChain
from langchain.chat_models import ChatOpenAI

from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from dotenv import load_dotenv


load_dotenv()


class DataTemplates:
    """Class for storing the templates for the different generation tasks."""

    def get_ten_dim_prompt(self) -> ChatPromptTemplate:
        system_message = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=[],
                template="""
        You are an advanced AI writer. Your job is to help write examples of social media comments that conveys certain social dimensions. The social dimensions are: social support, conflict, trust, neutral, fun, respect, knowledge, power, and similarity/identity.
        """,
            )
        )

        human_message = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=[
                    "social_dimension",
                    "social_dimension_description",
                    "text",
                ],
                template="""The following social media text conveys the social dimension {social_dimension}. {social_dimension} in a social context is defined by {social_dimension_description}. Write 10 new semantically similar examples in style of a social media comment, that show the same intent and social dimension.

Text: {text}



Answer:
""",
            )
        )
        return ChatPromptTemplate.from_messages([system_message, human_message])

    def get_sentiment_prompt(self) -> ChatPromptTemplate:
        system_message = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=[],
                template="""
        You are an advanced classifying AI. You are tasked with classifying the sentiment of a text. Sentiment can be either positive, negative or neutral.
        """,
            )
        )

        human_message = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=["sentiment", "text"],
                template="""Based on the following social media text which has a {sentiment} sentiment, write 10 new similar examples in style of a social media comment, that has the same sentiment. Separate the texts by newline.

Text: {text}

Answer:
""",
            )
        )
        return ChatPromptTemplate.from_messages([system_message, human_message])

    def get_hate_speech_prompt(self) -> ChatPromptTemplate:
        system_message = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=[],
                template="""
        You are a helpful undergrad. Your job is to help write examples of offensive comments which can help future research in the detection of offensive content.
        """,
            )
        )

        human_message = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=["hate_speech", "text"],
                template="""Based on the following social media text which is {hate_speech} , write 10 new similar examples in style of a social media comment, that has the same sentiment. 
                Answer in Danish. 

Text: {text}

Answer:
""",
            )
        )
        return ChatPromptTemplate.from_messages([system_message, human_message])

    def get_alpaca_input_prompt(self) -> PromptTemplate:
        input_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

        ### Instruction:
        Rewrite the following {language} text in 5 different ways{label}. The answer must be in {language}.

        ### Input:
        {text}

        ### Response:"""

        return PromptTemplate(
            input_variables=["text", "language", "label"], template=input_template
        )


class ClassificationTemplates:
    """Class for storing the templates for the different classification tasks."""

    def classify_hate_speech(self) -> ChatPromptTemplate:
        system_message = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=[],
                template="""
        You are an advanced classifying AI. You are tasked with classifying whether a text is offensive or not.
        """,
            )
        )

        human_message = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=["text"],
                template="""The following is a comment on a social media post. Classify whether the post is offensive (OFF) or not (NOT). Your answer must be one of ["OFF", "NOT"].

Text: {text}

Answer:
""",
            )
        )
        return ChatPromptTemplate.from_messages([system_message, human_message])

    def classify_sentiment(self) -> ChatPromptTemplate:
        system_message = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=[],
                template="""
        You are an advanced classifying AI. You are tasked with classifying the sentiment of a text. Sentiment can be either positive, negative or neutral.
        """,
            )
        )

        human_message = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=["text"],
                template="""Classify the following social media comment into either “negative”, “neutral” or “positive”. Your answer MUST be either one of ["negative", "neutral", "positive"]. Your answer must be lowercased.

Text: {text}

Answer:
""",
            )
        )
        return ChatPromptTemplate.from_messages([system_message, human_message])

    def classify_ten_dim(self) -> ChatPromptTemplate:
        """Work in progress"""
        system_message = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=[],
                template="""
        You are an advanced classifying AI. You are tasked with classifying the social dimension of a text. The social dimensions are: social support, conflict, trust, neutral, fun, respect, knowledge, power, and similarity/identity.
        """,
            )
        )

        human_message = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=[
                    "text",
                ],
                template="""Based on the following social media text, classify the social dimension of the text. You answer MUST only be one of the social dimensions. Your answer MUST be exactly one of ["social_support", "conflict", "trust", "neutral", "fun", "respect", "knowledge", "power", "similarity_identity"]. The answer must be lowercased.

Text: {text}

Answer:
""",
            )
        )
        return ChatPromptTemplate.from_messages([system_message, human_message])


if __name__ == "__main__":
    # ten_dim_template = DataTemplates().get_ten_dim_prompt()

    classify_ten_dim = ClassificationTemplates().classify_hate_speech()

    # print(
    #     classify_ten_dim.format(
    #         text="Happy 22nd Birthday to the cuddy Peyton Siva aka PEY PEY!! #FumatuBloodline #AllStar #GoLouisville"
    #     )
    # )

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    llm_chain = LLMChain(prompt=classify_ten_dim, llm=llm)

    social_dimension = "social support"
    social_dimension_description = (
        " Giving emotional or practical aid and companionship"
    )
    text = [
        "Fucking h\u00f8jr\u00f8vet t\u00e5be. Det var da USA'S st\u00f8rste fejl at v\u00e6lge det fjols som president, hold k\u00e6ft en tegneserie figur",
        "Sidst jeg k\u00f8bte en flaske af det lort var efter de begyndte at bruge stevia istedet for gode gammeldags kemikalier, s\u00e5 min gule saftevand smagte af lakrids og var derfor udrikkeligt.  Jeg k\u00f8ber aldrig mere Fun, n\u00e5r det skal v\u00e6re p\u00e5 den m\u00e5de.",
    ]

    for i in text:
        # output = llm_chain.run(
        #     {
        #         "text": i,
        #         "social_dimension": social_dimension,
        #         "social_dimension_description": social_dimension_description,
        #     }
        # )
        output = llm_chain.run(
            {
                "text": i,
            }
        )
        print(f"Input: {i}")
        print(f"Output: {output}")
        print("-------")

    a = 1
