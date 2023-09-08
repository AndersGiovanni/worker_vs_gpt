import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset

DATASET_NAME = "crowdflower"
DATA_DIR = "/Users/arpe/Documents/ITU/COCOONS/worker_vs_gpt/data/"+DATASET_NAME

def main():

    data_hf = load_dataset('Blablablab/SOCKET', DATASET_NAME)

    # test dataset
    test = pd.DataFrame()
    test["text"] = data_hf["test"]["text"]
    test["label"] = data_hf["test"]["label"]
    with open(DATA_DIR+"/data/label_list.txt", "r") as f:
        lines = f.readlines()
        id2label = {}
        for i in range(len(lines)):
            line = lines[i].strip()
            id2label[i] = line
    test["target"] = test["label"].map(lambda x: id2label[x])
    test.drop(columns=["label"], inplace=True)
    test.to_json(DATA_DIR+"/test.json", orient="records")
    
    # train dataset
    train_old = pd.DataFrame()
    train_old["text"] = data_hf["train"]["text"]
    train_old["label"] = data_hf["train"]["label"]

    # validation dataset
    validation = pd.DataFrame()
    validation["text"] = data_hf["validation"]["text"]
    validation["label"] = data_hf["validation"]["label"]

    train = pd.DataFrame()
    train["text"] = pd.concat(
        [train_old, validation])["text"]
    train["label"] = pd.concat(
        [train_old, validation])["label"]
    train["target"] = train["label"].map(lambda x: id2label[x])
    train.drop(columns=["label"], inplace=True)

    # cut train if size is more than 5000
    if train.shape[0] > 5000:
        train = train.sample(n=5000, random_state=42) # not stratified, to resemble the original dataset

    # get base dataset
    train, base = train_test_split(train, test_size=int(train.shape[0]*0.10), stratify=train["target"])

    train.to_json(DATA_DIR+"/train.json", orient="records")
    base.to_json(DATA_DIR+"/base.json", orient="records")


if __name__ == "__main__":
    main()

