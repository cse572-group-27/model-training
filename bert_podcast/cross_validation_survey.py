import yaml
from dataset import Dataset, SplitDataset
from model import BertClassifier
from util import evaluate,train
import json
from sklearn.model_selection import KFold, train_test_split
if __name__ == "__main__":
    config = yaml.safe_load(open("params.yaml","r"))
    csv_path = config["csv_path"]
    labels = config["labels"]
    epochs = config["train_parms"]["epochs_total"]

    lr = 1e-6
    kf = KFold(5)
    
    begin_ds = Dataset(csv_path, labels)
    last_ds = Dataset(csv_path, labels, first=False)

    indices = range(len(begin_ds))
    train_idx, test_idx = train_test_split(indices, train_size=0.8, test_size=0.2, random_state=400)
    begin_ds = SplitDataset(begin_ds, train_idx)
    last_ds = SplitDataset(last_ds, train_idx)
    
    # train on first tokens
    print("On first 512 tokens:")
    for i, (train_index, val_index) in enumerate(kf.split(begin_ds)):
        model = BertClassifier()
        train_ds = SplitDataset(begin_ds,train_index)
        val_ds = SplitDataset(begin_ds,val_index)
        train_data = train(model, train_ds, val_ds, lr, epochs)
        with open(f'logs/data_first_{i}.json', 'w') as f:
            json.dump(train_data, f)
        
    # train on last tokens
    print("On last 512 tokens:")
    for i, (train_index, val_index) in enumerate(kf.split(last_ds)):
        model = BertClassifier()
        train_ds = SplitDataset(begin_ds,train_index)
        val_ds = SplitDataset(begin_ds,val_index)
        train_data = train(model, train_ds, val_ds, lr, epochs)
        with open(f'logs/data_last_{i}.json', 'w') as f:
            json.dump(train_data, f)
    