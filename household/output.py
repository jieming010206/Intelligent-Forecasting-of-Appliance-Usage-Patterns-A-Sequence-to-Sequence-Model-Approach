import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from pretreatment import load_data_2

from datasets import Datasets
from sklearn.metrics import accuracy_score
from utils import save_evaluate, Custom_metric
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TIME_LABELS = [f"E{i}" for i in range(144)]
APPLIANCE_LABELS = ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10"]
results = []
models = ["5e-6_lstm_0.3", "7e-6_lstm_0.3", "lstm_0.3", "2e-5_lstm_0.3", "5e-5_lstm_0.3"]
for model_name in models:
    if model_name == "base_lstm":
        from models.base_lstm import Model
    elif model_name == "base_gru":
        from models.base_gru import Model
    elif model_name == "lstm" or model_name == "lstm_0.5" or model_name == "lstm_0.3" or model_name == "lstm_0.1":
        from models.lstm import Model
    elif model_name == "gru":
        from models.gru import Model
    elif model_name == "lstm_large":
        from models.lstm import Model
    elif model_name == "lstm_att":
        from models.lstm_att import Model
    else:
        from models.lstm import Model
    MODEL_PATH = f"outputs/{model_name}/model.pkl"
    model = Model(144, 11, 12)  # define model structure
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))  # load model 
    model = model.to(DEVICE)  # use CPU/GPU
    model.eval()  # model evaluation
    
    X_train, X_val, y_train, y_val = load_data_2(60, 12)
    test_datasets = Datasets(X_val, y_val)  # Assuming Datasets class is defined properly
        
    test_loader = DataLoader(
        test_datasets,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )  # training dataloader

    time_real_targets = []  
    time_pred_targets = []
    appliance_real_targets = []  
    appliance_pred_targets = []  
    day_real_targets = []  
    day_pred_targets = [] 
    test_loss_records = []
    custom_metric_records = []
    
    for idx, batch_data in enumerate(test_loader):  
        inputs, targets = batch_data  

        time_outputs, appliance_outputs = model(inputs.to(DEVICE), targets.to(DEVICE))  # 前向传播
    
        time_targets = targets[:,:,0].reshape(-1)
        appliance_targets = targets[:, :, 1].reshape(-1)
        time_outputs = time_outputs.reshape(-1, 144)
        appliance_outputs = appliance_outputs.reshape(-1, 11)
            
        time_real_targets.extend(time_targets.tolist())  # record appliance targets
        time_pred_targets.extend(torch.argmax(time_outputs, dim=1).cpu().tolist()) 
        appliance_real_targets.extend(appliance_targets.tolist())  # record appliance targets
        appliance_pred_targets.extend(torch.argmax(appliance_outputs, dim=1).cpu().tolist())  # record appliance prediction

    custom_metric = Custom_metric(time_real_targets, time_pred_targets, appliance_real_targets, appliance_pred_targets)
    time_train_acc = round(accuracy_score(time_real_targets, time_pred_targets), 4)
    appliance_train_acc = round(accuracy_score(appliance_real_targets, appliance_pred_targets), 4)  # calculate appliance acc
    val_acc = round((time_train_acc + appliance_train_acc ) / 2, 4)  # mean accuracy
    time_lcs, time_edit_distance, time_hamming_distance = save_evaluate(
                  time_real_targets, time_pred_targets, TIME_LABELS, None
              )
    appliance_lcs, appliance_edit_distance, appliance_hamming_distance = save_evaluate(
                  appliance_real_targets, appliance_pred_targets, APPLIANCE_LABELS, None
              )
    
    results.append({
        "Model": model_name,
        "Custom Metric": custom_metric,
        "Accuracy": val_acc,
        "Time LCS": time_lcs,
        "Time Edit Distance": time_edit_distance,
        "Appliance LCS": appliance_lcs,
        "Appliance Edit Distance": appliance_edit_distance
    })

df = pd.DataFrame(results)
df.to_csv('result.csv', index=False)

print(df)