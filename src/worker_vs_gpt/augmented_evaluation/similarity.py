# https://stackoverflow.com/questions/65199011/is-there-a-way-to-check-similarity-between-two-full-sentences-in-python

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict

import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
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
