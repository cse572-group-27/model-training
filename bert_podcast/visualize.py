import seaborn as sns

# Apply the default theme
sns.set_theme()
import matplotlib.pyplot as plt 
import json

prec = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
rec = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
for j in range(5):
    data = json.load(open(f"logs/data_last_{j}.json"))
    # print(data)
    for i in range(20):
        prec[i] += data[f"epoch_{i+1}"]["val_precision"]/5
        rec[i] += data[f"epoch_{i+1}"]["val_recall"]/5
prec[0] = 0
plt.title("Last 512 token survey")
plt.plot(list(range(1,21)),prec,label="Val Precision")
plt.plot(list(range(1,21)),rec, label="Val Recall")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.legend()
plt.savefig("vis_last.png")
