from torch.optim import Adam
from tqdm import tqdm
import numpy as np
from torch.utils.data import WeightedRandomSampler
import torch
from sklearn.utils import class_weight
from torch import nn
import json
from collections import defaultdict

def test_two_models(model1, model2, dataset1, dataset2, device):
  tp,fp,tn,fn = 0.0,0.0,0.0,0.0
  total = 0.0
  data_loader1 = torch.utils.data.DataLoader(dataset1,shuffle=False,batch_size=1)
  data_loader2 = torch.utils.data.DataLoader(dataset2,shuffle=False,batch_size=1)
  for (inp,lbl),(inp2,lbl2) in zip(data_loader1,data_loader2):
    lbl = lbl.to(device)#.unsqueeze(0)
    mask = inp['attention_mask'].to(device)
    input_id = inp['input_ids'].squeeze(1).to(device)
    output = model1(input_id, mask)
    lbl2 = lbl2.to(device)#.unsqueeze(0)
    mask = inp2['attention_mask'].to(device)
    input_id = inp2['input_ids'].squeeze(1).to(device)
    output += model2(input_id, mask)
    # output = output.view(lbl.shape)
    predict = output.argmax(dim=1)
    total += len(lbl)
    tp += (lbl*(lbl==predict)).sum()
    tn += ((1-lbl)*(lbl==predict)).sum()
    fp += ((1-lbl)*(lbl!=predict)).sum()
    fn += (lbl*(lbl!=predict)).sum()
  return {"accuracy":(tp+tn)/(tp+tn+fp+fn), "precision":tp/(tp+fp), "recall": tp/(tp+fn)}

def evaluate(model, data_loader, device):
  tp,fp,tn,fn = 0.0,0.0,0.0,0.0
  total = 0.0
  for inp,lbl in data_loader:
    lbl = lbl.to(device)#.unsqueeze(0)
    mask = inp['attention_mask'].to(device)
    input_id = inp['input_ids'].squeeze(1).to(device)
    output = model(input_id, mask)
    # output = output.view(lbl.shape)
    predict = output.argmax(dim=1)
    total += len(lbl)
    tp += (lbl*(lbl==predict)).sum()
    tn += ((1-lbl)*(lbl==predict)).sum()
    fp += ((1-lbl)*(lbl!=predict)).sum()
    fn += (lbl*(lbl!=predict)).sum()
  return {"accuracy":(tp+tn)/(tp+tn+fp+fn), "precision":tp/(tp+fp), "recall": tp/(tp+fn)}
  # "True Positive: {:.2f}; False Positive: {:.2f}\nFalse Negative: {:.2f}; True Negative: {:.2f}\nPrecision: {:.2f}; Recall: {:.2f}".format(tp,fp,fn,tn,tp/(tp+fp),tp/(tp+fn))

def train(model, train_ds, val, learning_rate, epochs):
    class_counts = torch.tensor([train_ds.labels.count(label) for label in range(2)])
    num_samples = int(sum(class_counts))
    weights = [1.0 / class_counts[train_ds[i][-1].item()] for i in range(num_samples)]
    sampler = WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)
    train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=2, sampler=sampler)
    if val is not None:
      val_dataloader = torch.utils.data.DataLoader(val, batch_size=1)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    #nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr= learning_rate)
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_ds.labels), y=train_ds.labels)
    criterion = nn.CrossEntropyLoss(weight=torch.Tensor(class_weights))
    if use_cuda:
      model = model.cuda()
      criterion = criterion.cuda()
    data = defaultdict(lambda :{"train_loss":0,"train_precision":0, "train_recall":0,"train_accuracy":0, "val_precision":0,"val_recall":0,"val_accuracy":0})
    for epoch_num in range(epochs):
      # total_acc_train = 0
      total_loss_train = 0
      tp,fp,tn,fn = 0.0,0.0,0.0,0.0
      total = 0.0
      for train_input, train_label in tqdm(train_dataloader):
          train_label = train_label.to(device)
          mask = train_input['attention_mask'].to(device)
          input_id = train_input['input_ids'].squeeze(1).to(device)

          output = model(input_id, mask)
          batch_loss = criterion(output,train_label.long())
          total_loss_train += batch_loss.item()
          predict = output.argmax(dim=1)
          total += len(train_label)
          tp += (train_label*(train_label==predict)).sum()
          tn += ((1-train_label)*(train_label==predict)).sum()
          fp += ((1-train_label)*(train_label!=predict)).sum()
          fn += (train_label*(train_label!=predict)).sum()

          model.zero_grad()
          batch_loss.backward()
          optimizer.step()

      if val is not None:
          with torch.no_grad():
              test_result = evaluate(model, val_dataloader, device)
              data[f"epoch_{epoch_num+1}"]["val_accuracy"] = float(test_result["accuracy"])
              data[f"epoch_{epoch_num+1}"]["val_precision"] = float(test_result["precision"])
              data[f"epoch_{epoch_num+1}"]["val_recall"] = float(test_result["recall"])
      data[f"epoch_{epoch_num+1}"]["train_loss"] = float(total_loss_train / len(train_ds))
      data[f"epoch_{epoch_num+1}"]["train_precision"] = float(tp/(fp+tp))
      data[f"epoch_{epoch_num+1}"]["train_recall"] = float(tp/(tp+fn))
      data[f"epoch_{epoch_num+1}"]["train_accuracy"] = float((tp+tn)/(tp+fp+tn+fn))
      # print(
      #     f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_ds): .3f} \
      #     \nTrain Precision: {tp/(tp+fp):.2f}; Recall: {tp/(tp+fn):.2f}\n\
      #     Validation set result:\n')# \
          
    return data
