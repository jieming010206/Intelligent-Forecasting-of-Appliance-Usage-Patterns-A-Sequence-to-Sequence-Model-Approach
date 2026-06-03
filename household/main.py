from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from configs import MODEL_NAME, INPUT_SEQ_LEN, OUTPUT_SEQ_LEN, DAY_LABELS, TIME_LABELS, APPLIANCE_LABELS, DEVICE, CONFIG
from datasets import Datasets
from pretreatment import load_data, load_merge_data, load_data_2, load_merge_data_2
from train import train

import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger('matplotlib.font_manager').disabled = True



if MODEL_NAME == "base_lstm":
    from models.base_lstm import Model
elif MODEL_NAME == "base_gru":
    from models.base_gru import Model
elif MODEL_NAME == "lstm" or MODEL_NAME == "lstm_0.5" or MODEL_NAME == "lstm_0.3" or MODEL_NAME == "lstm_0.1":
    from models.lstm import Model
elif MODEL_NAME == "gru":
    from models.gru import Model
elif MODEL_NAME == "lstm_large":
    from models.lstm import Model
elif MODEL_NAME == "lstm_att":
    from models.lstm_att import Model
else:
    from models.lstm import Model


def get_label_weights(names, labels, values):
    # label weights
    counts = [count for label, count in sorted(Counter(values).most_common(), key=lambda x: x[0])]
    label_ratios = dict(zip(labels, [round(count / sum(counts), 4) for count in counts]))
    print(f"{names} label ratios: {label_ratios}")
    weights = [sum(counts) / count for count in counts]
    weights = [round(weight / sum(weights), 4) for weight in weights]
    label_weights = dict(zip(labels, weights))
    print(f"{names} label weights: {label_weights}")
    print(len(labels))
    return weights


def train_run(X_train, X_val, y_train, y_val):
    train_datasets = Datasets(X_train, y_train)  # training set
    val_datasets = Datasets(X_val, y_val)  # validation set

    train_loader = DataLoader(
        train_datasets,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )  # training dataloader
    val_loader = DataLoader(
        val_datasets,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )  # validation dataloader

    model = Model(len(TIME_LABELS), len(APPLIANCE_LABELS), OUTPUT_SEQ_LEN) # Model
    model = model.to(DEVICE)  # CPU/GPU
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])  # optimizer
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=0.5, patience=CONFIG["patience"] // 2, verbose=True
    )  # learning rate scheduler
    
    time_criterion = torch.nn.CrossEntropyLoss(
        weight=torch.FloatTensor(
            get_label_weights(
                "time",
                TIME_LABELS,
                np.array(y_train)[:, :, 0].flatten().tolist(),
            )
        ).to(DEVICE)
    ).to(DEVICE)
    time_criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
    appliance_criterion = torch.nn.CrossEntropyLoss(
        weight=torch.FloatTensor(
            get_label_weights(
                "APPLIANCE",
                APPLIANCE_LABELS,
                np.array(y_train)[:, :, 1].flatten().tolist(),
            )
        ).to(DEVICE)
    )  # appliance category loss function


    train(train_loader, val_loader, model, optimizer, scheduler, time_criterion, appliance_criterion)  # 开始训练


if __name__ == "__main__":
    print(CONFIG)
    print(MODEL_NAME)
    X_train, X_val, Y_train, Y_val = load_data_2(INPUT_SEQ_LEN, OUTPUT_SEQ_LEN)
    print(np.shape(X_train))
    print(np.shape(X_val))
    print(np.shape(Y_train))
    print(np.shape(Y_val))
    train_run(X_train, X_val, Y_train, Y_val)  # model training
