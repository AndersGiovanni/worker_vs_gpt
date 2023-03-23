import pandas as pd
from worker_vs_gpt.config import TEN_DIM_DATA_DIR
from sklearn.model_selection import train_test_split


def main():
    # load a json file from ten-dim to a pandas dataframe
    data = pd.read_json(TEN_DIM_DATA_DIR / "labeled_dataset.json", orient="records")
    # drop the columns that are not needed

    # convert the labels to binary
    labels = [
        "social_support",
        "conflict",
        "trust",
        "fun",
        "similarity",
        "identity",
        "respect",
        "knowledge",
        "power",
        "other",
        "romance",
    ]
    for label in labels:
        data[label] = data[label].map(lambda x: 1 if x >= 2 else 0)

    # find neutral comments
    data["neutral"] = (data[labels].sum(axis=1) == 0).astype(int)
    data = data.drop(columns=["other", "romance"])

    # combine similarity and identity
    data["similarity_identity"] = data.apply(
        lambda x: 1 if x["similarity"] == 1 or x["identity"] == 1 else 0, axis=1
    )
    data = data.drop(columns=["similarity", "identity"])

    # new labels
    labels = [
        "social_support",
        "conflict",
        "trust",
        "neutral",
        "fun",
        "respect",
        "knowledge",
        "power",
        "similarity_identity",
    ]

    # convert to multiclass
    data = data.melt(
        id_vars=["text", "h_text", "round"], var_name="target", value_vars=labels
    )
    data = data[data["value"] == 1]
    data = data.drop(columns=["value", "round"])

    # shuffle the data
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    # Select 20% for test
    train, test = train_test_split(data, test_size=0.2, stratify=data["target"])

    # Make stratified train and test splits with size 500
    train, base = train_test_split(train, test_size=500, stratify=train["target"])

    train.to_json(TEN_DIM_DATA_DIR / "train.json", orient="records")
    base.to_json(TEN_DIM_DATA_DIR / "base.json", orient="records")
    test.to_json(TEN_DIM_DATA_DIR / "test.json", orient="records")

    print("Done")


if __name__ == "__main__":
    main()
