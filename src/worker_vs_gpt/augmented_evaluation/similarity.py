# https://stackoverflow.com/questions/65199011/is-there-a-way-to-check-similarity-between-two-full-sentences-in-python

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict

import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from rouge import Rouge
from sacrebleu.metrics import BLEU
from tqdm import tqdm
from utils import assert_path
from utils import load_json
from utils import save_json


nltk.download("punkt")
nltk.download("stopwords")


@dataclass()
class SimilarityScorer:
    def __init__(self, spacy_model: str = "en_core_web_lg"):
        self.spacy_model = spacy.load(spacy_model)
        self.stop_words = set(stopwords.words("english"))
        self.rouge_scorer = Rouge()
        self.bleu_scorer = BLEU(effective_order=True)

    def spacy_embedding_similarity(self, text1: str, text2: str) -> float:
        """
        Calculates the similarity between two texts using spacy's word embeddings.
        Similarity is the same whether text1 or text2 is the original text.
        """
        doc_text1 = self.spacy_model(text1)
        doc_text2 = self.spacy_model(text2)
        return doc_text1.similarity(doc_text2)

    def vocab_overlap(self, str1: str, str2: str) -> Dict:
        # Tokenize the input strings
        tokens1 = set(word_tokenize(str1))
        tokens2 = set(word_tokenize(str2))

        # Remove common English stopwords
        tokens1 = tokens1 - self.stop_words
        tokens2 = tokens2 - self.stop_words

        # Calculate the token overlap
        overlap = len(tokens1.intersection(tokens2))

        # Identify new words
        new_words1 = tokens1 - tokens2
        new_words2 = tokens2 - tokens1

        # Percentage overlap
        percentage_overlap = overlap / (len(tokens1) + len(tokens2))

        return {
            "token_overlap": overlap,
            "percentage_token_overlap": percentage_overlap,
            "new_words_in_str1": list(new_words1),
            "new_words_in_str2": list(new_words2),
        }

    def bleu_score(self, hypothesis: str, reference: str) -> float:
        """
        Calculates the BLEU score between two strings.
        """

        score = self.bleu_scorer.sentence_score(
            hypothesis=hypothesis,
            references=[reference],
        )

        return score.score / 100  # sacreBLEU gives the score in percent

    def rouge_score(self, hypothesis: str, reference: str) -> float:
        score = self.rouge_scorer.get_scores(
            hyps=hypothesis,
            refs=reference,
        )
        return score[0]["rouge-l"]["f"]


if __name__ == "__main__":
    pass
