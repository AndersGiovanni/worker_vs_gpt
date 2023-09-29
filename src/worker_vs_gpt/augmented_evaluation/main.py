""" 
The initial script for combining the original text and the augmented text and evaluating the results with multiple metrics.
hatespeech er den eneste som kan indeholde flere sprog
"""


if __name__ == "__main__":
    import json
    from collections import defaultdict
    from typing import Dict

    import spacy
    from similarity import SimilarityScorer
    from tqdm import tqdm
    from utils import assert_path
    from utils import load_json
    from utils import save_json

    SS = SimilarityScorer()

    dataset_name: str = "hayati_politeness"
    assert_path(f"results/{dataset_name}/")

    # Results dictionary
    results_cosine: Dict = defaultdict(list)
    results_vocab_overlap: Dict = defaultdict(list)

    # open the data
    filename: str = f"../../../data/{dataset_name}/balanced_gpt-4_augmented.json"
    data: Dict = load_json(filename)

    # load the spacy model
    for text_entries in tqdm(data):
        target: str = text_entries["target"]
        text: str = text_entries["text"]
        augmented_text: str = text_entries["augmented_text"]

        similarity: float = SS.spacy_embedding_similarity(text, augmented_text)
        results_cosine[target].append(similarity)

        vocab_overlap: Dict = SS.vocab_overlap(text, augmented_text)
        results_vocab_overlap[target].append(vocab_overlap)

    save_json(results_cosine, f"results/{dataset_name}/spacy_similarity.json")
    save_json(results_vocab_overlap, f"results/{dataset_name}/vocab_overlap.json")
