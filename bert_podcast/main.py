import yaml
from dataset import Dataset, SplitDataset
from model import BertClassifier
from util import evaluate,train
from sklearn.model_selection import KFold, train_test_split
if __name__ == "__main__":
    config = yaml.safe_load(open("params.yaml","r"))
    csv_path = config["csv_path"]
    labels = config["labels"]
    epoch_first512 = config["train_parms"]["epoch_first512"]
    epoch_last512 = config["train_parms"]["epoch_last512"]
    lr = 1e-6
    kf = KFold(5)
    
    begin_ds = Dataset(csv_path, labels)
    last_ds = Dataset(csv_path, labels, first=False)

    indices = range(len(begin_ds))
    train_idx, test_idx = train_test_split(indices, train_size=0.8, test_size=0.2, random_state=400)
    begin_ds = SplitDataset(begin_ds, train_idx)
    last_ds = SplitDataset(last_ds, train_idx)
    
    # train on first tokens
    print("Training model on first 512 tokens:")
    model_first = BertClassifier()
    train(model_first, begin_ds, None, lr, epoch_first512)
    torch.save(model_first.state_dict(),"first512.pth")
    # train on last tokens
    print("Training model on last 512 tokens:")
    model_last = BertClassifier()
    train(model_last, begin_ds, None, lr, epoch_first512)
    torch.save(model_last.state_dict(),"last512.pth")
    