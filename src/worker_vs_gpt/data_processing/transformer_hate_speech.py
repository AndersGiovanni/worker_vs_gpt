import pandas as pd
from worker_vs_gpt.config import HATE_SPEECH_DATA_DIR
from sklearn.model_selection import train_test_split


def main():
    test = pd.read_csv(
        HATE_SPEECH_DATA_DIR / "oe20da_data/offenseval-da-test-v1.tsv",
        sep="\t",  # Read a txt file with a single column of text there is no separator
    )
    train = pd.read_csv(
        HATE_SPEECH_DATA_DIR / "oe20da_data/offenseval-da-training-v1.tsv",
        sep="\t",  # Read a txt file with a single column of text there is no separator
    )

    # rename subtask_a to target
    test.rename(columns={"subtask_a": "target"}, inplace=True)
    train.rename(columns={"subtask_a": "target"}, inplace=True)

    # drop id column
    test.drop(columns=["id"], inplace=True)
    train.drop(columns=["id"], inplace=True)

    # discard last row in train
    train = train[:-1]

    test.to_json(HATE_SPEECH_DATA_DIR / "test.json", orient="records")

    # Make stratified train and test splits with size 500
    train, base = train_test_split(train, test_size=500, stratify=train["target"])

    train.to_json(HATE_SPEECH_DATA_DIR / "train.json", orient="records")
    base.to_json(HATE_SPEECH_DATA_DIR / "base.json", orient="records")

    print("Done")


if __name__ == "__main__":
    main()
