import os
from dataclasses import dataclass
from typing import Dict, List

import backoff
import openai
from dotenv import load_dotenv
from openai.openai_object import OpenAIObject

from ten_social_dim.prompting.prompts import (
    ClassificationPrompts,
    DataAugmentationPrompts,
    PromptGenerator,
)

load_dotenv()


@dataclass
class OpenAIClient:
    """OpenAI Client."""

    def __init__(
        self, model: str = "text-davinci-003", method: str = "augment"
    ) -> None:
        """Init."""
        self.model: str = model
        self.method = method  # can be 'augment' or 'predict'

        if self.method == "predict":
            self.prompter: PromptGenerator = ClassificationPrompts(
                labels=[
                    "knowledge",
                    "power",
                    "respect",
                    "trust",
                    "social_support",
                    "romance",
                    "similarity",
                    "identity",
                    "fun",
                    "conflict",
                ]
            )
        elif self.method == "augment":
            self.prompter: PromptGenerator = DataAugmentationPrompts()
        else:
            raise NotImplementedError(f"{self.method} not implemented.")

        # Set API key
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def generate(
        self,
        input_text: str,
        h_text: str,
        max_tokens: int = 2500,
        temperature: float = 0.0,
    ) -> OpenAIObject:
        """Generate text."""
        if self.model in ["text-davinci-003"]:
            prompt: str = DataAugmentationPrompts().gpt_35(
                input_text=input_text, h_text=h_text
            )
            resp: OpenAIObject = self.text_completion(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        elif self.model in ["gpt-3.5-turbo"]:
            prompt: List[Dict[str, str]] = DataAugmentationPrompts().chat_gpt(
                input_text=input_text, h_text=h_text
            )
            resp: OpenAIObject = self.chat_completion(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        else:
            raise NotImplementedError(f"{self.model} not implemented.")

        return resp

    @backoff.on_exception(backoff.expo, openai.error.RateLimitError)
    def predict(
        self, input_text: str, max_tokens: int = 2800, temperature: float = 0.0
    ) -> OpenAIObject:
        """Predict text."""

        # GPT-3.5
        if self.model in ["text-davinci-003"]:
            prompt_gpt_35: str = self.prompter.gpt_35(
                input_text=input_text, h_text=None
            )

            return self.text_completion(
                prompt=prompt_gpt_35,
                max_tokens=max_tokens,
                temperature=temperature,
            )

        # ChatGPT
        elif self.model in ["gpt-3.5-turbo"]:
            prompt_chat_gpt: List[Dict[str, str]] = self.prompter.chat_gpt(
                input_text=input_text, h_text=None
            )

            return self.chat_completion(
                prompt=prompt_chat_gpt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        else:
            raise NotImplementedError(f"{self.model} not implemented.")

    def text_completion(
        self,
        prompt: str,
        max_tokens: int = 50,
        temperature: float = 0.0,
    ) -> OpenAIObject:
        """Text completion. GPT-3.5."""

        resp: OpenAIObject = openai.Completion.create(
            model=self.model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp

    def chat_completion(
        self,
        prompt: List[Dict[str, str]],
        max_tokens: int = 50,
        temperature: float = 0.0,
    ) -> OpenAIObject:
        """Chat completion. ChatGPT."""

        resp: OpenAIObject = openai.ChatCompletion.create(
            model=self.model,
            messages=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp
