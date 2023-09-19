import os
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
import pandas as pd

from worker_vs_gpt.config import POLITENESS_DATA_DIR, SAMESIDE_DATA_DIR
from worker_vs_gpt.utils import few_shot_sampling


load_dotenv()


class DataTemplates:
    """Class for storing the templates for the different generation tasks."""

    def get_crowdfl_prompt(self) -> ChatPromptTemplate:
        system_message = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=[],
                template="""
        You are an advanced AI writer. Your job is to help write examples of social media comments that convey certain emotions. Emotions to be considered are: sadness, enthusiasm, empty, neutral, worry, love, fun, hate, happiness, relief, boredom, surprise, anger.
        """,
            )
        )

        human_message = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=["text", "emotion"],
                template="""The following social media text conveys the emotion {emotion}. Write 9 new semantically similar examples in the style of a social media comment, that show the same intent and emotion.

Text: {text}



Answer:
""",
            )
        )
        return ChatPromptTemplate.from_messages([system_message, human_message])

    def get_hypo_prompt(self) -> ChatPromptTemplate:
        system_message = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=[],
                template="""
        You are an advanced AI writer. You are tasked with writing examples of sentences that are hyperbolic or not.
        """,
            )
        )

        human_message = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=["text", "hypo"],
                template="""The following sentence has a {hypo} flag for being hyperbolic. Write 9 new semantically similar examples that show the same intent and hyperbolic flag.

Text: {text}



Answer:
""",
            )
        )
        return ChatPromptTemplate.from_messages([system_message, human_message])

    def get_sameside_prompt(self) -> ChatPromptTemplate:
        system_message = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=[],
                template="""
        You are an advanced AI writer. Your job is to help write examples of sentence pairs, separated by the [SEP] tag, that convey the same stance or not. 
        """,
            )
        )

        human_message = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=["text", "side"],
                template="""The following sentence pair has a {side} flag for showing stances that are on the same side. Write 9 new semantically similar examples of sentence pairs in the style of online debate sites arguments, that show the same intent and same side stance flag.

Text: {text}



Answer:
""",
            )
        )
        return ChatPromptTemplate.from_messages([system_message, human_message])

    def get_empathy_prompt(self) -> ChatPromptTemplate:
        system_message = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=[],
                template="""
        You are an advanced AI writer. Your job is to help write examples of texts that convey empathy or not.
        """,
            )
        )

        human_message = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=["text", "empathy"],
                template="""The following text has a {empathy} flag for expressing empathy, write 9 new semantically similar examples that show the same intent and empathy flag.

Text: {text}



Answer:
""",
            )
        )
        return ChatPromptTemplate.from_messages([system_message, human_message])

    def get_intimacy_prompt(self) -> ChatPromptTemplate:
        system_message = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=[],
                template="""
        You are an advanced AI writer. Your job is to help write examples of question posted in social media that convey certain levels of intimacy. The intimacy levels are: very intimate, intimate, somewhat intimate, not very intimate, not intimate, not intimate at all.
        """,
            )
        )

        human_message = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=["text", "intimacy"],
                template="""The following social media question conveys the {intimacy} level of question intimacy. Write 9 new semantically similar examples in the style of a social media question, that show the same intent and intimacy level.

Text: {text}



Answer:
""",
            )
        )
        return ChatPromptTemplate.from_messages([system_message, human_message])

    def get_talkdown_prompt(self) -> ChatPromptTemplate:
        system_message = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=[],
                template="""
        You are an advanced AI writer. Your job is to help write examples of social media comments that convey condescendence or not. 
        """,
            )
        )

        human_message = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=["text", "talkdown"],
                template="""The following social media text has a {talkdown} flag for showing condescendence, write 9 new semantically similar examples in the style of a social media comment, that show the same intent and condescendence flag.

Text: {text}



Answer:
""",
            )
        )
        return ChatPromptTemplate.from_messages([system_message, human_message])

    def get_politeness_prompt(self) -> ChatPromptTemplate:
        system_message = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=[],
                template="""
        You are an advanced AI writer. Your job is to help write examples of social media comments that convey politeness or not.
        """,
            )
        )

        human_message = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=["text", "politeness"],
                template="""The following social media text has a {politeness} flag for politeness, write 9 new semantically similar examples in the style of a social media comment, that show the same intent and politeness flag.

Text: {text}



Answer:
""",
            )
        )
        return ChatPromptTemplate.from_messages([system_message, human_message])

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
                template="""The following social media text conveys the social dimension {social_dimension}. {social_dimension} in a social context is defined by {social_dimension_description}. Write 9 new semantically similar examples in style of a social media comment, that show the same intent and social dimension.

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
                template="""Based on the following social media text which has a {sentiment} sentiment, write 9 new similar examples in style of a social media comment, that has the same sentiment. Separate the texts by newline.

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
                template="""Based on the following social media text which is {hate_speech} , write 9 new similar examples in style of a social media comment, that has the same sentiment. 
                Answer in Danish. 

Text: {text}

Answer:
""",
            )
        )
        return ChatPromptTemplate.from_messages([system_message, human_message])


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
                input_variables=["few_shot", "text"],
                template="""The following is a comment on a social media post. Classify whether the post is offensive (OFF) or not (NOT). Your answer must be one of ["OFF", "NOT"].
{few_shot}
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
                input_variables=["few_shot", "text"],
                template="""Classify the following social media comment into either “negative”, “neutral” or “positive”. Your answer MUST be either one of ["negative", "neutral", "positive"]. Your answer must be lowercased.
{few_shot}
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
                    "few_shot",
                    "text",
                ],
                template="""Based on the following social media text, classify the social dimension of the text. You answer MUST only be one of the social dimensions. Your answer MUST be exactly one of ["social_support", "conflict", "trust", "neutral", "fun", "respect", "knowledge", "power", "similarity_identity"]. The answer must be lowercased.
{few_shot}
Text: {text}

Answer:
""",
            )
        )
        return ChatPromptTemplate.from_messages([system_message, human_message])

    def classify_crowdflower(self) -> ChatPromptTemplate:
        """Work in progress"""
        system_message = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=[],
                template="""
        You are an advanced classifying AI. You are tasked with classifying the emotion of a text. The emotions are: sadness, enthusiasm, empty, neutral, worry, love, fun, hate, happiness, relief, boredom, surprise, anger.
        """,
            )
        )

        human_message = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=[
                    "few_shot",
                    "text",
                ],
                template="""Based on the following social media text, classify the emotion of the text. You answer MUST only be one of the emotions. Your answer MUST be exactly one of ['sadness', 'enthusiasm', 'empty', 'neutral', 'worry', 'love', 'fun', 'hate', 'happiness', 'relief', 'boredom', 'surprise', 'anger']. The answer must be lowercased.
{few_shot}
Text: {text}

Answer:
""",
            )
        )
        return ChatPromptTemplate.from_messages([system_message, human_message])

    def classfify_same_side(self) -> ChatPromptTemplate:
        system_message = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=[],
                template="""
        You are an advanced classifying AI. You are tasked with classifying the whether two texts, separated by [SEP], convey the same stance or not. The two stances are 'not same side' and 'same side'.
        """,
            )
        )

        human_message = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=[
                    "few_shot",
                    "text",
                ],
                template="""Based on the following text, classify the stance of the text. You answer MUST only be one of the stances. Your answer MUST be exactly one of ['not same side', 'same side']. The answer must be lowercased.
{few_shot}
Text: {text}
Answer:
""",
            )
        )

        return ChatPromptTemplate.from_messages([system_message, human_message])

    def classfify_hypo(self) -> ChatPromptTemplate:
        system_message = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=[],
                template="""
        You are an advanced classifying AI. You are tasked with classifying the whether the text is a hyperbole or not a hyperbole.
        """,
            )
        )

        human_message = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=["few_shot", "text"],
                template="""Based on the following text, classify the text is a hyperbole. You answer MUST only be one of the two labels. Your answer MUST be exactly one of ['hyperbole', 'not hyperbole']. The answer must be lowercased.
{few_shot}
Text: {text}

Answer:
""",
            )
        )
        return ChatPromptTemplate.from_messages([system_message, human_message])

    def classfify_hayati(self) -> ChatPromptTemplate:
        system_message = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=[],
                template="""
        You are an advanced classifying AI. You are tasked with classifying the whether the text is polite or impolite.
        """,
            )
        )

        human_message = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=["few_shot", "text"],
                template="""Based on the following text, classify the politeness of the text. You answer MUST only be one of the two labels. Your answer MUST be exactly one of ['impolite', 'polite']. The answer must be lowercased.
{few_shot}
Text: {text}

Answer:
""",
            )
        )
        return ChatPromptTemplate.from_messages([system_message, human_message])

    def classfify_empathy(self) -> ChatPromptTemplate:
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
        return ChatPromptTemplate.from_messages([system_message, human_message])

    def classfify_intimacy(self) -> ChatPromptTemplate:
        system_message = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=[],
                template="""
        You are an advanced classifying AI. You are tasked with classifying the intimacy of the text. The different intimacies are 'Very intimate', 'Intimate', 'Somewhat intimate', 'Not very intimate', 'Not intimate', and 'Not intimate at all'.
        """,
            )
        )

        human_message = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=["few_shot", "text"],
                template="""Based on the following text, classify how intimate the text is. You answer MUST only be one of the six labels. Your answer MUST be exactly one of ['Very-intimate', 'Intimate', 'Somewhat-intimate', 'Not-very-intimate', 'Not-intimate', 'Not-intimate-at-all'].
{few_shot}
Text: {text}

Answer:
""",
            )
        )
        return ChatPromptTemplate.from_messages([system_message, human_message])

    def classify_talkdown(self) -> ChatPromptTemplate:
        system_message = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=[],
                template="""
        You are an advanced classifying AI. You are tasked with classifying if the text is condescending or not condescending.
        """,
            )
        )

        human_message = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=["few_shot", "text"],
                template="""Based on the following text, classify if it is condescending. You answer MUST only be one of the two labels. Your answer MUST be exactly one of ['not condescension', 'condescension'].
{few_shot}
Text: {text}

Answer:
""",
            )
        )
        return ChatPromptTemplate.from_messages([system_message, human_message])


if __name__ == "__main__":
    # ten_dim_template = DataTemplates().get_ten_dim_prompt()

    text = "text"  # text column (can be text or h_text)
    test = pd.read_json(os.path.join(POLITENESS_DATA_DIR, "test.json"))
    train = pd.read_json(os.path.join(POLITENESS_DATA_DIR, "train.json"))
    classification_prompt = ClassificationTemplates().classfify_hayati()

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    llm_chain = LLMChain(prompt=classification_prompt, llm=llm, verbose=True)

    for i in test[text]:
        few_shot_samples = few_shot_sampling(train, 2)

        output = llm_chain.run({"text": i, "few_shot": few_shot_samples})

        # p = classification_prompt.format({"text": i, "few_shot": few_shot_samples})

        print(output)
        print("-------")

    a = 1
