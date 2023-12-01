import torch
from transformers import BertTokenizer
import pandas as pd
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, labels, first=True):
        df = pd.read_csv(csv_path)
        self.labels = [labels[label] for label in df['label']]
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        if first:
          self.texts = [self.tokenizer(str(text),
                               padding='max_length', max_length = 512, truncation=True,
                                  return_tensors="pt") for text in df['text'].values]
        else:
          self.texts = [self.tokenizer(" ".join(str(text).split()[-512:]),
                               padding='max_length', max_length = 512, truncation=True,
                                  return_tensors="pt") for text in df['text'].values]
    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y

class SplitDataset(torch.utils.data.Dataset):
    def __init__(self,dataset, split):
        self.labels = [dataset.labels[s] for s in split]
        self.texts = [dataset.texts[s] for s in split]
    def classes(self):
        return self.labels
    def __len__(self):
        return len(self.labels)
    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y