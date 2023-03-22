import pandas as pd
from worker_vs_gpt.config import SENTIMENT_DATA_DIR


def main():
    # read in mapping.txt as id2label dictionary
    with open(SENTIMENT_DATA_DIR / "data/mapping.txt", "r") as f:
        lines = f.readlines()
        id2label = {}
        for line in lines:
            line = line.strip()
            line = line.split("\t")
            id2label[line[0]] = line[1]

    test = pd.DataFrame()
    test["text"] = pd.read_csv(
        SENTIMENT_DATA_DIR / "data/test_text.txt",
        sep="\n\n",  # Read a txt file with a single column of text there is no separator
        header=None,
        names=["text"],
    )["text"]
    test["label"] = pd.read_csv(
        SENTIMENT_DATA_DIR / "data/test_labels.txt",
        sep="\n\n",
        header=None,
        names=["label"],
    )["label"]
    test["label"] = test["label"].map(lambda x: id2label[str(x)])

    test.to_json(SENTIMENT_DATA_DIR / "test.json", orient="records")

    train = pd.DataFrame()
    train["text"] = pd.concat(
        [
            pd.read_csv(
                SENTIMENT_DATA_DIR / "data/val_text.txt",
                sep="\n\n",
                header=None,
                names=["text"],
            ),
            pd.read_csv(
                SENTIMENT_DATA_DIR / "data/train_text.txt",
                sep="\n\n",
                header=None,
                names=["text"],
            ),
        ]
    )["text"]

    train["label"] = pd.concat(
        [
            pd.read_csv(
                SENTIMENT_DATA_DIR / "data/val_labels.txt",
                sep="\n\n",
                header=None,
                names=["label"],
            ),
            pd.read_csv(
                SENTIMENT_DATA_DIR / "data/train_labels.txt",
                sep="\n\n",
                header=None,
                names=["label"],
            ),
        ]
    )["label"]

    train["label"] = train["label"].map(lambda x: id2label[str(x)])

    train.to_json(SENTIMENT_DATA_DIR / "train.json", orient="records")

    print("Done")


if __name__ == "__main__":
    main()
