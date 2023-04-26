import pytest
from worker_vs_gpt.utils import LabelMatcher, parse_output


@pytest.fixture
def label_matcher_ten_dim():
    labels = [
        "conflict",
        "social_support",
        "fun",
        "power",
        "neutral",
        "knowledge",
        "respect",
        "trust",
        "similarity_identity",
    ]
    return LabelMatcher(labels, task="ten-dim")


@pytest.fixture
def label_matcher_hate_speech():
    labels = ["OFF", "NOT"]
    return LabelMatcher(labels, task="hate-speech")


@pytest.fixture
def label_matcher_sentiment():
    labels = ["positive", "negative", "neutral"]
    return LabelMatcher(labels, task="sentiment")


def test_match_exact(label_matcher_ten_dim):
    label = "respect"
    text = "I have a lot of respect for you."
    assert label_matcher_ten_dim(label, text) == "respect"


def test_match_case_insensitive(label_matcher_ten_dim):
    label = "CONFLICT"
    text = "There is some conflict between us."
    assert label_matcher_ten_dim(label, text) == "conflict"


def test_match_partial(label_matcher_ten_dim):
    label = "identity"
    text = "I identify strongly with that group."
    assert label_matcher_ten_dim(label, text) == "similarity_identity"


def test_match_unknown_label(label_matcher_ten_dim):
    label = "happiness"
    text = "I feel happy today."
    assert label_matcher_ten_dim(label, text) == "neutral"


def test_task_not_found():
    with pytest.raises(ValueError):
        LabelMatcher(["conflict", "social_support"], task="emotion")


def test_hate_speech_task(label_matcher_hate_speech):
    label = "OFF"
    text = "This is hate speech."
    assert label_matcher_hate_speech(label, text) == "OFF"


def test_hate_speech_task(label_matcher_hate_speech):
    label = "NOT"
    text = "This is not hate speech."
    assert label_matcher_hate_speech(label, text) == "NOT"


def test_hate_speech_task(label_matcher_hate_speech):
    label = "NOT_EXISTING"
    text = "This is not hate speech."
    assert label_matcher_hate_speech(label, text) == "NOT"


def test_sentiment_task(label_matcher_sentiment):
    label = "positive"
    text = "I love this product."
    assert label_matcher_sentiment(label, text) == "positive"


# Test the parse_output function


def test_parse_output_single_line():
    input_string = "1. Item one"
    expected_output = ["Item one"]
    assert parse_output(input_string) == expected_output


def test_parse_output_multiple_lines():
    input_string = "1. Item one\n2. Item two\n3. Item three"
    expected_output = ["Item one", "Item two", "Item three"]
    assert parse_output(input_string) == expected_output


def test_parse_output_leading_whitespace():
    input_string = "1.     Item one\n2.    Item two\n3.    Item three"
    expected_output = ["Item one", "Item two", "Item three"]
    assert parse_output(input_string) == expected_output


def test_parse_output_empty_items():
    input_string = "1. Item one\n2.    \n3. Item three\n4.  \n5. Item five"
    expected_output = ["Item one", "Item three", "Item five"]
    assert parse_output(input_string) == expected_output


def test_parse_output_no_items():
    input_string = ""
    expected_output = []
    assert parse_output(input_string) == expected_output


def test_parse_output_high_enumerations():
    input_string = "1. Item one\n2.    \n3. Item three\n4.  \n521. Item five"
    expected_output = ["Item one", "Item three", "Item five"]
    assert parse_output(input_string) == expected_output
