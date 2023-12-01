from util import test_two_models
from model import BertClassifier
from dataset import Dataset, SplitDataset
from sklearn.model_selection import KFold, train_test_split
import yaml
import torch
if __name__ == "__main__":
    config = yaml.safe_load(open("params.yaml","r"))
    csv_path = config["csv_path"]
    labels = config["labels"]
    begin_ds = Dataset(csv_path, labels)
    last_ds = Dataset(csv_path, labels, first=False)
    indices = range(len(begin_ds))
    train_idx, test_idx = train_test_split(indices, train_size=0.8, test_size=0.2, random_state=400)
    begin_ds = SplitDataset(begin_ds, test_idx)
    last_ds = SplitDataset(last_ds, test_idx)
    model1 = BertClassifier()
    model2 = BertClassifier()
    model1.load_state_dict(torch.load("first512.pth"))
    model2.load_state_dict(torch.load("last512.pth"))
    model1.to("cuda")
    model2.to("cuda")
    model2.eval()
    model2.eval()
    print(test_two_models(model1,model2,begin_ds,last_ds,"cuda"))
