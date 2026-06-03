import warnings
from difflib import SequenceMatcher

import dill
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import hamming
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

from configs import (
    APPLIANCE_LABELS,
    MODEL_PATH,
    ACC_RECORDER_PATH,
    ACC_VISUALIZATION_PATH,
    LOSS_RECORDER_PATH,
    LOSS_VISUALIZATION_PATH,
    APPLIANCE_EVALUATE_PATH,
    DURATION_EVALUATE_PATH,
    APPLIANCE_CONFUSION_MATRIX_PATH,
    DURATION_CONFUSION_MATRIX_PATH,
    DEVICE,
    CONFIG,
    OUTPUT_SEQ_LEN,
)

warnings.filterwarnings("ignore")  # ignore warning
plt.rcParams["font.sans-serif"] = ["SimHei"]  
plt.rcParams["axes.unicode_minus"] = False 


def save_pkl(filepath, data):

    with open(filepath, "wb") as fw:
        dill.dump(data, fw)
    print(f"[{filepath}] data saving...")


def load_pkl(filepath):

    with open(filepath, "rb") as fr:
        data = dill.load(fr, encoding="utf-8")
    print(f"[{filepath}] data loading...")
    return data


def save_txt(filepath, data):
    with open(filepath, "w", encoding="utf-8") as fw:
        fw.write(data)
    print(f"{filepath} saving...")


def get_lcs(s1, s2):
    n = len(s1)
    lcs_len = [[0] * (n + 1) for _ in range(2)]
    current, previous = 0, 1
    
    for i in range(1, n + 1):
        current, previous = previous, current
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                lcs_len[current][j] = lcs_len[previous][j - 1] + 1
            else:
                lcs_len[current][j] = max(lcs_len[previous][j], lcs_len[current][j - 1])
    return lcs_len[current][n]

def get_edit_distance(s1, s2):
    d = [[x for x in range(len(s1) + 1)] for _ in range(len(s2) + 1)]
    for y in range(1, len(s2) + 1):
        d[y][0] = d[y - 1][0] + 1
    for x in range(1, len(s1) + 1):
        for y in range(1, len(s2) + 1):
            if s1[x - 1] == s2[y - 1]:
                d[y][x] = d[y - 1][x - 1]
            else:
                substute = d[y - 1][x - 1] + 1
                add = d[y][x - 1] + 1
                delete = d[y - 1][x] + 1
                d[y][x] = min(add, substute, delete)
    return d[-1][-1]



def get_hamming_distance(s1, s2):
    return int(hamming(s1, s2) * len(s1))

def Custom_metric(t1, t2, a1, a2):
    t1 = np.array(t1)
    t2 = np.array(t2)
    a1 = np.array(a1)
    a2 = np.array(a2)
    
    t = (t1 == t2).astype(int)
    a = (a1 == a2).astype(int)
    return round(np.dot(a, t)/len(t), 4)
    
def save_evaluate(y_val, y_val_pred, labels, output_path):
    report = classification_report(
        y_val,
        y_val_pred,
        labels=range(0, len(labels)),
        target_names=labels,
        digits=4,
        zero_division=0,
    )  # performance metrics including precision/recall/f1-score/accuracy
    matrix = confusion_matrix(y_val, y_val_pred)  # compute confusion matrix

    lcs_recorder = []
    edit_distance_recorder = []
    hamming_distance_recorder = []
    for seq1, seq2 in zip(np.array(y_val).reshape(-1, OUTPUT_SEQ_LEN), np.array(y_val_pred).reshape(-1, OUTPUT_SEQ_LEN)):
        lcs_recorder.append(get_lcs(seq1, seq2))  # LCS
        edit_distance_recorder.append(get_edit_distance(seq1, seq2))  # edite distance
        hamming_distance_recorder.append(get_hamming_distance(seq1, seq2))  # hamming distance
    avg_lcs = round(np.mean(lcs_recorder).item(), 4) # average lcs
    avg_edit_distance = round(np.mean(edit_distance_recorder).item(), 4)  # average edite distance
    avg_hamming_distance = round(np.mean(hamming_distance_recorder).item(), 4)  # average hamming distance

    evaluate = (
        "Classification Report:\n"
        + report
        + "\n\nConfusion Matrix:\n"
        + str(matrix)
        + "\n\nAverage LCS: "
        + str(avg_lcs)
        + "\nAverage Edit Distance: "
        + str(avg_edit_distance)
        + "\nAverage Hamming Distance: "
        + str(avg_hamming_distance)
        + "\n"
    )

    # save_txt(output_path, evaluate)
    return avg_lcs, avg_edit_distance, avg_hamming_distance
    # print(evaluate)  


def plot_confusion_matrix(y_val, y_val_pred, labels, output_path):
    matrix = confusion_matrix(y_val, y_val_pred) 
    matrix = matrix.astype("float") / matrix.sum(axis=1)[:, np.newaxis]  

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.4, 4.8), dpi=100) 
    sns.heatmap(matrix, annot=True, fmt=".2f", linewidths=0.5, square=True, cmap="Blues", ax=ax) 
    ax.set_title("Confusion Matrix")  
    ax.set_xlabel("Actual label")  
    ax.set_ylabel("Predicted label") 
    ax.set_xticks([i + 0.5 for i in range(len(labels))]) 
    ax.set_yticks([i + 0.5 for i in range(len(labels))])  
    ax.set_xticklabels(labels, rotation=0)
    ax.set_yticklabels(labels, rotation=0)
    plt.savefig(output_path) 
    plt.close()  
  


def epoch_visualization(y1, y2, name, output_path):
    plt.figure(figsize=(16, 9), dpi=100)  
    plt.plot(y1, marker=".", linestyle="-", linewidth=2, label=f"train {name}") 
    plt.plot(y2, marker=".", linestyle="-", linewidth=2, label=f"test {name}")  
    plt.title(f"{name} change map during training", fontsize=24) 
    plt.xlabel("epoch", fontsize=20)  
    plt.ylabel(name, fontsize=20) 
    plt.tick_params(labelsize=16)  
    plt.legend(loc="best", prop={"size": 20})  
    plt.savefig(output_path) 
    # plt.show()  
    plt.close()  

