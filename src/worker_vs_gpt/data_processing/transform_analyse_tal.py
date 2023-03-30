from worker_vs_gpt.config import ANALYSE_TAL_DATA_DIR
from sklearn.model_selection import train_test_split

import pandas as pd


if __name__ == "__main__":
    train = pd.read_csv(ANALYSE_TAL_DATA_DIR / "train.csv", sep="\t", header=None)
    eval = pd.read_csv(ANALYSE_TAL_DATA_DIR / "eval.csv", sep="\t", header=None)
    test = pd.read_csv(ANALYSE_TAL_DATA_DIR / "test.csv", sep="\t", header=None)

    # combine train and eval
    train = pd.concat([train, eval])

    # Select column index 1 and 2
    train = train[[1, 2]]
    test = test[[1, 2]]

    # give columns names
    train.columns = ["target", "text"]
    test.columns = ["target", "text"]

    test.to_json(ANALYSE_TAL_DATA_DIR / "test.json", orient="records")

    # Shuffle and reset index
    train = train.sample(frac=1, random_state=42).reset_index(drop=True)

    train, base = train_test_split(train, test_size=500, stratify=train["target"])

    train.to_json(ANALYSE_TAL_DATA_DIR / "train.json", orient="records")
    base.to_json(ANALYSE_TAL_DATA_DIR / "base.json", orient="records")
