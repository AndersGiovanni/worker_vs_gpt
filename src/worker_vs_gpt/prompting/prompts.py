from dataclasses import dataclass

from typing import Dict, Tuple, Union, List


@dataclass
class PromptGenerator:
    """PromptGenerator."""

    def gpt_35(self, input_text: str, h_text: Union[str, None]) -> str:
        """Create prompt for GPT-3.5.
        The functionality is implemented in the subclasses."""
        raise NotImplementedError

    def chat_gpt(
        self, input_text: str, h_text: Union[str, None]
    ) -> List[Dict[str, str]]:
        """Create prompt for ChatGPT.
        The functionality is implemented in the subclasses."""
        raise NotImplementedError


@dataclass
class DataAugmentationPrompts(PromptGenerator):
    """Data augmentation prompts."""

    def gpt_35(self, input_text: str, h_text: Union[str, None]) -> str:
        """Create data augmentation prompt for GPT-3.5."""

        instruction: str = """Suppose the following texts are based on an online
        conversation on a social media platform. Based on the
        following snippet of text, create new text snippets containing
        the same intent, but written differently. Create also a h_text,
        which is an exact highlight of the new text snippet."""

        prompt: str = f"""{instruction}
        Original text: {input_text}
        h_text: {h_text}
        New text:"""

        return prompt

    def chat_gpt(
        self, input_text: str, h_text: Union[str, None]
    ) -> List[Dict[str, str]]:
        """Create data augmentation prompt for ChatGPT."""

        return [
            {
                "role": "user",
                "content": "Suppose the following texts are based on an online conversation on a social media platform.",
            },
            {
                "role": "user",
                "content": f"text: {input_text}",
            },
            {
                "role": "user",
                "content": f"This is a highlight of the text, h_text: {h_text}",
            },
            {
                "role": "user",
                "content": "Generate a new text based on the original text and h_text. The new texts should have the same intent as 'text', but written differently. Generate also a h_text, which is an exact highlight of the new text. text:",
            },
        ]


@dataclass
class ClassificationPrompts(PromptGenerator):
    """Classification prompts."""

    def __init__(self, labels: List[str]) -> None:
        self.labels: List[str] = labels

        # Based on the selected labels, initialize label explanation and keywords
        self.label_description, self.label_keywords = self._init_label_explanation()

    def chat_gpt(
        self, input_text: str, h_text: Union[str, None] = None
    ) -> List[Dict[str, str]]:
        string_descriptions, string_keywords = "", ""

        for label in self.labels:
            string_descriptions += f"{label}: {self.label_description[label]} \n"
            string_keywords += f"{label}: {self.label_keywords[label]} \n"

        return [
            {
                "role": "system",
                "content": "You are a classification model. You are given a text snippet and asked to classify it into one or more social dimensions. The social dimensions are:",
            },
            {
                "role": "system",
                "content": "knowledge, power, respect, trust, social_support, romance, similarity, identity, fun, conflict.",
            },
            {
                "role": "system",
                "content": "You are given the following descriptions and keywords of social dimensions.",
            },
            {"role": "system", "content": f"Descriptions: {string_descriptions}"},
            {"role": "system", "content": f"Keywords: {string_keywords}"},
            {
                "role": "user",
                "content": f"Given the following text snippet, please select the social dimensions that best describe the text. Report the label(s) in the following format: label1, label2\nText: {input_text}\nLabels:",
            },
        ]

    def gpt_35(self, input_text: str, h_text: Union[str, None] = None) -> str:
        string_descriptions, string_keywords = "", ""

        for label in self.labels:
            string_descriptions += f"{label}: {self.label_description[label]} \n"
            string_keywords += f"{label}: {self.label_keywords[label]} \n"

        prompt: str = f"""Given the following descriptions and keywords of social dimensions, please select the social dimensions that best describes the following text snippet. Separate the labels with a comma.
        
        Descriptions: 
        {string_descriptions}
        
        Keywords: 
        {string_keywords}
        
        Text: {input_text}
        Labels:
        """
        return prompt

    def _init_label_explanation(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Initialize label explanation and keywords based on the list of labels."""
        label_description: Dict[str, str] = {
            "knowledge": "Exchange of ideas or information; learning, teaching",
            "power": "Having power over the behavior and outcomes of another",
            "respect": "Conferring status, appreciation, gratitude, or admiration upon another",
            "trust": "Will of relying on the actions or judgments of another",
            "social_support": "Giving emotional or practical aid and companionship",
            "romance": "Intimacy among people with a sentimental or sexual relationship",
            "similarity": "Shared interests, motivations or outlooks",
            "identity": "Shared sense of belonging to the same community or group",
            "fun": "Experiencing leisure, laughter, and joy",
            "conflict": "Contrast or diverging views",
        }

        label_keywords: Dict[str, str] = {
            "knowledge": "teaching, intelligence, competent, expertise, know-how, insight",
            "power": "command, control, dominance, authority, pretentious, decisions",
            "respect": "admiration, appreciation, praise, thankful, respect, honor",
            "trust": "trustworthy, honest, reliable, dependability, loyalty, faith",
            "social_support": "friendly, caring, cordial, sympathy, companionship, encouragement",
            "romance": "love, sexual, intimacy, partnership, affection, emotional, couple",
            "similarity": "alike, compatible, equal, congenial, affinity, agreement",
            "identity": "community, united, identity, cohesive, integrated",
            "fun": "funny, humor, playful, comedy, cheer, enjoy, entertaining",
            "conflict": "hatred, mistrust, tense, disappointing, betrayal, hostile",
        }

        return {label: label_description[label] for label in self.labels}, {
            label: label_keywords[label] for label in self.labels
        }
