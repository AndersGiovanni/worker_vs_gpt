"""
Iterate over all the datasets in the data folder and calculate the similarity 
metrics between the original and the augmented text.
"""


from pathlib import Path
from typing import List

import nltk
import torch
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sentence_transformers import util as sentence_transformer_util
from tqdm import tqdm


class TransformerSimilarity:
    def __init__(
        self, sentence_transformer_model_name: str = "intfloat/e5-base"
    ) -> None:
        self.device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
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


def vocab_overlap(original: str, augmented: str) -> float:
    # Tokenize the input strings
    tokens1 = set(word_tokenize(original))
    tokens2 = set(word_tokenize(augmented))

    # Remove common English stopwords
    tokens1 = tokens1 - set(stopwords.words("english"))
    tokens2 = tokens2 - set(stopwords.words("english"))

    # Calculate the token overlap
    overlap = len(tokens1.intersection(tokens2))

    # # Identify new words
    # new_words1 = tokens1 - tokens2
    # new_words2 = tokens2 - tokens1

    # Percentage overlap
    percentage_overlap = overlap / (len(tokens1) + len(tokens2))

    return percentage_overlap


if __name__ == "__main__":
    from pathlib import Path

    from worker_vs_gpt.utils import read_json
    from worker_vs_gpt.utils import save_json

    outputdir = Path("src/worker_vs_gpt/evaluation/semantic-evaluation")
    datadir = Path("data")

    for dataset in datadir.iterdir():
        # two files in each dataset:
        # - data/{dataset}/balanced_gpt-4_augmented.json
        # - data/{dataset}/balanced_llama-2-70b_augmented.json

        # create a new folder for each dataset
        dataset_outputdir = outputdir / dataset.name

        # create the folder if it doesn't exist
        if not dataset_outputdir.exists():
            dataset_outputdir.mkdir()

        for augmented_file in dataset.iterdir():
            output_file = dataset_outputdir / augmented_file.name

            if output_file.exists():
                print(f"Skipping {output_file}")
                continue

            if augmented_file.name.endswith(
                "augmented.json"
            ) and augmented_file.name.startswith("balanced_"):
                data = read_json(augmented_file)
                output_data = []

                # start the similarity scorer
                similarity = TransformerSimilarity()

                for example in tqdm(data):
                    original: str = example["text"]
                    augmented: str = example["augmented_text"]
                    target: str = example["target"]

                    # cosine similarity
                    similarity_score: float = similarity.embedding_similarity(
                        original, augmented
                    )

                    # token overlap excluding stopwords
                    token_overlap: float = vocab_overlap(original, augmented)

                    # add the data to the output
                    output_data.append(
                        {
                            "original": original,
                            "augmented": augmented,
                            "target": target,
                            "similarity_score": similarity_score,
                            "token_overlap": token_overlap,
                        }
                    )

                # write the output data to a file
                output_file = dataset_outputdir / augmented_file.name
                print(f"Writing {output_file}")
                save_json(path=output_file, container=output_data)
