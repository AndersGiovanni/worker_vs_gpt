# https://stackoverflow.com/questions/65199011/is-there-a-way-to-check-similarity-between-two-full-sentences-in-python

import datetime
import os
from dataclasses import dataclass
from typing import Dict
from typing import List

import nltk
import spacy
import torch
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from rouge import Rouge
from sacrebleu.metrics import BLEU
from sentence_transformers import SentenceTransformer
from sentence_transformers import util as sentence_transformer_util
from utils import assert_path
from utils import load_json
from utils import save_json


# nltk.download("punkt")
# nltk.download("stopwords")


# "intfloat_e5-base"
class TransformerSimilarity:
    def __init__(
        self, sentence_transformer_model_name: str = "intfloat/e5-base"
    ) -> None:
        self.device = torch.device("mps" if torch.has_mps else "cpu")
        self.sentence_transformer_model = self.model = SentenceTransformer(
            sentence_transformer_model_name
        ).to(self.device)

    def embedding_similarity(self, text1: str, text2: str) -> float:
        text1_embedding: torch.tensor = self.model.encode(
            self._text_to_sentences(text1), convert_to_tensor=True
        )
        text2_embedding: torch.tensor = self.model.encode(
            self._text_to_sentences(text2), convert_to_tensor=True
        )

        cosine_scores = sentence_transformer_util.pytorch_cos_sim(
            text1_embedding, text2_embedding
        )
        return cosine_scores.mean().item()

    def _text_to_sentences(self, text: str) -> List[str]:
        return nltk.sent_tokenize(text)


class SimilarityScorer:
    def __init__(
        self,
        spacy_model_name: str = "en_core_web_lg",
    ):
        self.spacy_model = spacy.load(spacy_model_name)
        self.stop_words = set(stopwords.words("english"))
        self.rouge_scorer = Rouge()
        self.bleu_scorer = BLEU(effective_order=True)

    def spacy_cosine_similarity(self, text1: str, text2: str) -> float:
        """
        Calculates the similarity between two texts using spacy's word embeddings.
        Similarity is the same whether text1 or text2 is the original text.
        """
        doc_text1 = self.spacy_model(text1)
        doc_text2 = self.spacy_model(text2)
        return doc_text1.similarity(doc_text2)

    def vocab_overlap(self, original: str, augmented: str) -> Dict:
        # Tokenize the input strings
        tokens1 = set(word_tokenize(original))
        tokens2 = set(word_tokenize(augmented))

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
            "new_words_in_original": list(new_words1),
            "new_words_in_augmented": list(new_words2),
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
        """
        Calculates the ROUGE score between two strings.
        """
        score = self.rouge_scorer.get_scores(
            hyps=hypothesis,
            refs=reference,
        )
        return score[0]["rouge-l"]["f"]


@dataclass
class TextPair:
    original: str
    augmented: str
    target: str
    scorer: SimilarityScorer
    transformer_scorer: TransformerSimilarity

    @property
    def spacy_cosine_similarity(self) -> float:
        """
        Calculates the similarity between two texts using spacy's word embeddings.
        Similarity is the same whether text1 or text2 is the original text.
        """
        return self.scorer.spacy_cosine_similarity(self.original, self.augmented)

    @property
    def vocab_overlap(self) -> Dict:
        return self.scorer.vocab_overlap(self.original, self.augmented)

    @property
    def bleu_score(self) -> float:
        return self.scorer.bleu_score(self.original, self.augmented)

    @property
    def rouge_score(self) -> float:
        return self.scorer.rouge_score(self.original, self.augmented)

    @property
    def transformer_similarity(self) -> float:
        return self.transformer_scorer.embedding_similarity(
            self.original, self.augmented
        )


def calculate_metrics(dataset_name: str = "crowdflower") -> None:
    print(f"Started process for dataset: {dataset_name}")

    SS = SimilarityScorer()
    TS = TransformerSimilarity()

    assert_path(f"results/{dataset_name}/")

    # Results dictionary
    results: List[Dict] = []

    # open the data
    filename: str = f"../../../data/{dataset_name}/balanced_gpt-4_augmented.json"

    # skip if the file already exists
    if os.path.exists(f"results/{dataset_name}/similarity.json"):
        print(f"!!! Skipping {dataset_name} as the file already exists.")
        return

    text: str = "h_text" if dataset_name == "ten-dim" else "text"
    augmented_text: str = (
        "augmented_h_text" if dataset_name == "ten-dim" else "augmented_text"
    )

    data: Dict = load_json(filename, verbose=False)

    print(f"# Calculating metrics for {dataset_name}")
    for i, text_entry in enumerate(data):
        if i % 100 == 0:
            print(
                f"## {dataset_name} - {i} / {len(data)} @ {datetime.datetime.now().strftime('%H:%M:%S')}"
            )

        tp: TextPair = TextPair(
            original=text_entry[text],
            augmented=text_entry[augmented_text],
            target=text_entry["target"],
            scorer=SS,
            transformer_scorer=TS,
        )

        metrics = {
            "spacy_cosine_similarity": tp.spacy_cosine_similarity,
            "bleu_score": tp.bleu_score,
            "rouge_score": tp.rouge_score,
            "transformer_similarity": tp.transformer_similarity,
            "vocab_overlap": tp.vocab_overlap,
        }

        text_entry["metrics"] = metrics
        results.append(text_entry)

    save_json(results, f"results/{dataset_name}/similarity.json")
    return


if __name__ == "__main__":
    import multiprocessing
    import os

    datasets: List[str] = [
        dataset
        for dataset in os.listdir("../../../data/")
        if dataset not in [".DS_Store", "similarity_results", "hate-speech"]
    ]

    pool = multiprocessing.Pool(processes=4)
    # Use the pool to run the function in parallel with different parameters
    pool.map(calculate_metrics, datasets)

    # Close the pool to release resources
    pool.close()
    pool.join()
