from pytest import approx
from similarity import SimilarityScorer


def test_spacy_embedding_similarity():
    SS = SimilarityScorer()
    assert SS.spacy_embedding_similarity("hello", "hello") == 1.0


def test_vocab_overlap_1():
    SS = SimilarityScorer()
    result = SS.vocab_overlap("hello", "hello")
    assert result["token_overlap"] == 1
    assert result["new_words_in_str1"] == []
    assert result["new_words_in_str2"] == []


def test_vocab_overlap_2():
    SS = SimilarityScorer()
    result = SS.vocab_overlap("hello world", "hello earth")
    assert result["token_overlap"] == 1
    assert result["new_words_in_str1"] == ["world"]
    assert result["new_words_in_str2"] == ["earth"]


def test_vocab_overlap_3():
    SS = SimilarityScorer()
    result = SS.vocab_overlap("hello the world", "hello earth")
    assert result["token_overlap"] == 1
    assert result["new_words_in_str1"] == ["world"]
    assert result["new_words_in_str2"] == ["earth"]


def test_bleu_score():
    SS = SimilarityScorer()
    hypothesis = "to make people trustworthy you need to trust them"
    reference = "the way to make people trustworthy is to trust them"
    score: float = SS.bleu_score(hypothesis, reference)
    assert score == approx(0.386, abs=0.01)


def test_rouge_score():
    SS = SimilarityScorer()
    hypothesis = "to make people trustworthy you need to trust them"
    reference = "the way to make people trustworthy is to trust them"
    score: float = SS.rouge_score(hypothesis, reference)
    assert score == approx(0.705, abs=0.01)
