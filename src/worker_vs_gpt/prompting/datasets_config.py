from dataclasses import dataclass


@dataclass
class DatasetConfig:
    """Dataset config class."""

    name: str
    augmentation_system_prompt: str
    augmentation_task_prompt: str
    classification_system_prompt: str
    classification_task_prompt: str


HatespeechConfig = DatasetConfig(
    name="hate-speech",
    augmentation_system_prompt="You are a helpful undergrad. Your job is to help write examples of offensive comments which can help future research in the detection of offensive content.",
    augmentation_task_prompt="""Based on the following social media text which is {label} , write 9 new similar examples in style of a social media comment, that has the same sentiment. Answer in Danish. """,
    classification_system_prompt="You are an advanced classifying AI. You are tasked with classifying whether a text is offensive or not.",
    classification_task_prompt="The following is a comment on a social media post. Classify whether the post is offensive (OFF) or not (NOT). Your answer must be one of ['OFF', 'NOT'].",
)

SentimentConfig = DatasetConfig(
    name="sentiment",
    augmentation_system_prompt="You are an advanced AI writer. Your job is to help write examples of text with a certain sentiment. Sentiment can be either positive, negative or neutral.",
    augmentation_task_prompt="""Based on the following social media text which has a {label} sentiment, write 9 new similar examples in style of a social media comment, that has the same sentiment. Separate the texts by newline.""",
    classification_system_prompt="You are an advanced classifying AI. You are tasked with classifying the sentiment of a text. Sentiment can be either positive, negative or neutral.",
    classification_task_prompt="Classify the following social media comment into either “negative”, “neutral” or “positive”. Your answer MUST be either one of ['negative', 'neutral', 'positive']. Your answer must be lowercased.",
)

TenDimConfig = DatasetConfig(
    name="ten-dim",
    augmentation_system_prompt="You are an advanced AI writer. Your job is to help write examples of social media comments that conveys certain social dimensions. The social dimensions are: social support, conflict, trust, neutral, fun, respect, knowledge, power, and similarity/identity.",
    augmentation_task_prompt="""The following social media text conveys the social dimension {label}. {label} in a social context is defined by {social_dimension_description}. Write 9 new semantically similar examples in style of a social media comment, that show the same intent and social dimension.""",
    classification_system_prompt="You are an advanced classifying AI. You are tasked with classifying the social dimension of a text. The social dimensions are: social support, conflict, trust, neutral, fun, respect, knowledge, power, and similarity/identity.",
    classification_task_prompt="Based on the following social media text, classify the social dimension of the text. You answer MUST only be one of the social dimensions. Your answer MUST be exactly one of ['social_support', 'conflict', 'trust', 'neutral', 'fun', 'respect', 'knowledge', 'power', 'similarity_identity']. The answer must be lowercased.",
)

CrowdflowerConfig = DatasetConfig(
    name="crowdflower",
    augmentation_system_prompt="You are an advanced AI writer. Your job is to help write examples of social media comments that convey certain emotions. Emotions to be considered are: sadness, enthusiasm, empty, neutral, worry, love, fun, hate, happiness, relief, boredom, surprise, anger.",
    augmentation_task_prompt="""The following social media text conveys the emotion {label}. Write 9 new semantically similar examples in the style of a social media comment, that show the same intent and emotion.""",
    classification_system_prompt="You are an advanced classifying AI. You are tasked with classifying the emotion of a text. The emotions are: sadness, enthusiasm, empty, neutral, worry, love, fun, hate, happiness, relief, boredom, surprise, anger.",
    classification_task_prompt="Based on the following social media text, classify the emotion of the text. You answer MUST only be one of the emotions. Your answer MUST be exactly one of ['sadness', 'enthusiasm', 'empty', 'neutral', 'worry', 'love', 'fun', 'hate', 'happiness', 'relief', 'boredom', 'surprise', 'anger']. The answer must be lowercased.",
)

SameSidePairsConfig = DatasetConfig(
    name="same-side-pairs",
    augmentation_system_prompt="You are an advanced AI writer. Your job is to help write examples of sentence pairs, separated by the [SEP] tag, that convey the same stance or not. ",
    augmentation_task_prompt="""The following sentence pair has a {label} flag for showing stances that are on the same side. Write 9 new semantically similar examples of sentence pairs in the style of online debate sites arguments, that show the same intent and same side stance flag.""",
    classification_system_prompt="You are an advanced classifying AI. You are tasked with classifying the whether two texts, separated by [SEP], convey the same stance or not. The two stances are 'not same side' and 'same side'.",
    classification_task_prompt="Based on the following text, classify the stance of the text. You answer MUST only be one of the stances. Your answer MUST be exactly one of ['not same side', 'same side']. The answer must be lowercased.",
)

HayatiPolitenessConfig = DatasetConfig(
    name="hayati_politeness",
    augmentation_system_prompt="You are an advanced AI writer. Your job is to help write examples of social media comments that convey politeness or not.",
    augmentation_task_prompt="""The following social media text has a {label} flag for politeness, write 9 new semantically similar examples in the style of a social media comment, that show the same intent and politeness flag.""",
    classification_system_prompt="You are an advanced classifying AI. You are tasked with classifying the whether the text is polite or impolite.",
    classification_task_prompt="Based on the following text, classify the politeness of the text. You answer MUST only be one of the two labels. Your answer MUST be exactly one of ['impolite', 'polite']. The answer must be lowercased.",
)

HypoConfig = DatasetConfig(
    name="hypo-l",
    augmentation_system_prompt="You are an advanced AI writer. You are tasked with writing examples of sentences that are hyperbolic or not.",
    augmentation_task_prompt="""The following sentence has a {label} flag for being hyperbolic. Write 9 new semantically similar examples that show the same intent and hyperbolic flag.""",
    classification_system_prompt="You are an advanced classifying AI. You are tasked with classifying the whether the text is a hyperbole or not a hyperbole.",
    classification_task_prompt="Based on the following text, classify the text is a hyperbole. You answer MUST only be one of the two labels. Your answer MUST be exactly one of ['hyperbole', 'not hyperbole']. The answer must be lowercased.",
)

EmpathyConfig = DatasetConfig(
    name="empathy#empathy_bin",
    augmentation_system_prompt="You are an advanced AI writer. Your job is to help write examples of texts that convey empathy or not.",
    augmentation_task_prompt="""The following text has a {label} flag for expressing empathy, write 9 new semantically similar examples that show the same intent and empathy flag.""",
    classification_system_prompt="You are an advanced classifying AI. You are tasked with classifying the whether the text expresses empathy.",
    classification_task_prompt="Based on the following text, classify whether the text expresses empathy or not. You answer MUST only be one of the two labels. Your answer MUST be exactly one of ['empathy', 'not empathy']. The answer must be lowercased.",
)

QuestionIntimacyConfig = DatasetConfig(
    name="questionintimacy",
    augmentation_system_prompt="You are an advanced AI writer. Your job is to help write examples of question posted in social media that convey certain levels of intimacy. The intimacy levels are: very intimate, intimate, somewhat intimate, not very intimate, not intimate, not intimate at all.",
    augmentation_task_prompt="""The following social media question conveys the {label} level of question intimacy. Write 9 new semantically similar examples in the style of a social media question, that show the same intent and intimacy level.""",
    classification_system_prompt="You are an advanced classifying AI. You are tasked with classifying the intimacy of the text. The different intimacies are 'Very intimate', 'Intimate', 'Somewhat intimate', 'Not very intimate', 'Not intimate', and 'Not intimate at all'.",
    classification_task_prompt="Based on the following text, classify how intimate the text is. You answer MUST only be one of the six labels. Your answer MUST be exactly one of ['Very-intimate', 'Intimate', 'Somewhat-intimate', 'Not-very-intimate', 'Not-intimate', 'Not-intimate-at-all'].",
)

TalkdownPairsConfig = DatasetConfig(
    name="talkdown-pairs",
    augmentation_system_prompt="You are an advanced AI writer. Your job is to help write examples of social media comments that convey condescendence or not.",
    augmentation_task_prompt="""The following social media text has a {label} flag for showing condescendence, write 9 new semantically similar examples in the style of a social media comment, that show the same intent and condescendence flag.""",
    classification_system_prompt="You are an advanced classifying AI. You are tasked with classifying if the text is condescending or not condescending.",
    classification_task_prompt="Based on the following text, classify if it is condescending. You answer MUST only be one of the two labels. Your answer MUST be exactly one of ['not condescension', 'condescension'].",
)
