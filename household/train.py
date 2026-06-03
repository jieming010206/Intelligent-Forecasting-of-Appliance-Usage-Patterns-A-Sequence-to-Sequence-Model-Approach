import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from utils import get_edit_distance

from configs import (
    TIME_LABELS,
    APPLIANCE_LABELS,
    MODEL_PATH,
    ACC_RECORDER_PATH,
    ACC_VISUALIZATION_PATH,
    LOSS_RECORDER_PATH,
    LOSS_VISUALIZATION_PATH,
    TIME_EVALUATE_PATH,
    APPLIANCE_EVALUATE_PATH,
    DURATION_EVALUATE_PATH,
    TIME_CONFUSION_MATRIX_PATH,
    APPLIANCE_CONFUSION_MATRIX_PATH,
    DURATION_CONFUSION_MATRIX_PATH,
    DEVICE,
    CONFIG,
    OUTPUT_SEQ_LEN,
)
from utils import save_evaluate, plot_confusion_matrix, epoch_visualization

def get_edit_distance_for_subsequences(seq1, seq2, sub_length=OUTPUT_SEQ_LEN):
    results = []
    for i in range(0, len(seq1), sub_length):
        s1 = seq1[i:i+sub_length]
        s2 = seq2[i:i+sub_length]
        results.append(get_edit_distance(s1, s2))
    return np.mean(results)


def train_epoch(train_loader, model, optimizer, time_criterion, appliance_criterion, epoch):
    model.train()
    time_real_targets = []  
    time_pred_targets = []
    appliance_real_targets = []  
    appliance_pred_targets = []  
    day_real_targets = []  
    day_pred_targets = [] 
    train_loss_records = [] 
    train_distance_record = []
    for idx, batch_data in enumerate(tqdm(train_loader, file=sys.stdout)):  
        inputs, targets = batch_data

        time_outputs, appliance_outputs = model(inputs.to(DEVICE), targets.to(DEVICE))  # 前向传播
        
 
        time_targets = targets[:,:,0].reshape(-1)
        appliance_targets = targets[:, :, 1].reshape(-1)
        time_outputs = time_outputs.reshape(-1, len(TIME_LABELS))
        appliance_outputs = appliance_outputs.reshape(-1, len(APPLIANCE_LABELS))
        
        time_loss = time_criterion(time_outputs, time_targets.to(DEVICE))
        appliance_loss = appliance_criterion(appliance_outputs, appliance_targets.to(DEVICE)) 

        loss =  time_loss + appliance_loss   # sum of the loss
        optimizer.zero_grad()  
        loss.backward()  
        optimizer.step() 
        
        time_real_targets.extend(time_targets.tolist())  # record appliance targets
        time_pred_targets.extend(torch.argmax(time_outputs, dim=1).cpu().tolist()) 
        appliance_real_targets.extend(appliance_targets.tolist())  # record appliance targets
        appliance_pred_targets.extend(torch.argmax(appliance_outputs, dim=1).cpu().tolist())  # record appliance prediction
    
        train_loss_records.append(loss.item())  # record loss
        # distance = get_edit_distance_for_subsequences(appliance_real_targets, appliance_pred_targets)
        # train_distance_record.append(distance)


    time_train_acc = round(accuracy_score(time_real_targets, time_pred_targets), 4)
    appliance_train_acc = round(accuracy_score(appliance_real_targets, appliance_pred_targets), 4)  # calculate appliance acc

    train_acc = round((time_train_acc + appliance_train_acc) / 2, 4)  # mean accuracy
    train_loss = round(sum(train_loss_records) / len(train_loss_records), 4)  # mean loss
    # train_distance = round(sum(train_distance_record) / len(train_loss_records), 4)
    print(f"[train] Epoch: {epoch} / {CONFIG['epoch']}, acc: {train_acc}, loss: {train_loss}")
    return train_acc, train_loss


def evaluate(test_loader, model, scheduler, time_criterion, appliance_criterion, epoch):
    model.eval()
    time_real_targets = []  
    time_pred_targets = []
    appliance_real_targets = []  
    appliance_pred_targets = []  
    day_real_targets = []  
    day_pred_targets = [] 
    test_loss_records = []
    val_distance_records = []
    for idx, batch_data in enumerate(tqdm(test_loader, file=sys.stdout)):  
        inputs, targets = batch_data  

        time_outputs, appliance_outputs = model(inputs.to(DEVICE), targets.to(DEVICE))  # 前向传播

        
        time_targets = targets[:,:,0].reshape(-1)
        appliance_targets = targets[:, :, 1].reshape(-1)
        time_outputs = time_outputs.reshape(-1, len(TIME_LABELS))
        appliance_outputs = appliance_outputs.reshape(-1, len(APPLIANCE_LABELS))
        
        time_loss = time_criterion(time_outputs, time_targets.to(DEVICE))
        appliance_loss = appliance_criterion(appliance_outputs, appliance_targets.to(DEVICE)) 
 

        loss = time_loss + appliance_loss  # sum of the loss
        
    
        time_real_targets.extend(time_targets.tolist())  # record appliance targets
        time_pred_targets.extend(torch.argmax(time_outputs, dim=1).cpu().tolist()) 
        appliance_real_targets.extend(appliance_targets.tolist())  # record appliance targets
        appliance_pred_targets.extend(torch.argmax(appliance_outputs, dim=1).cpu().tolist())  # record appliance prediction
        test_loss_records.append(loss.item())  # record loss
        # distance = get_edit_distance_for_subsequences(appliance_real_targets, appliance_pred_targets)
        # val_distance_records.append(distance)


    time_train_acc = round(accuracy_score(time_real_targets, time_pred_targets), 4)
    appliance_train_acc = round(accuracy_score(appliance_real_targets, appliance_pred_targets), 4)  # calculate appliance acc
    val_acc = round((time_train_acc + appliance_train_acc ) / 2, 4)  # mean accuracy
    val_loss = round(sum(test_loss_records) / len(test_loss_records), 4)  # mean loss
    # val_distance = round(sum(val_distance_records) / len(val_distance_records), 4)
    print(f"[Eval] Epoch: {epoch} / {CONFIG['epoch']}, acc: {val_acc}, loss: {val_loss}")
    return (
        val_acc,
        val_loss,
        day_real_targets,
        day_pred_targets,
        time_real_targets,
        time_pred_targets,
        appliance_real_targets,
        appliance_pred_targets,
    )


def train(train_loader, test_loader, model, optimizer, scheduler, time_criterion, appliance_criterion):
    best_val_acc = 0   
    best_val_loss = np.inf   
    patience_counter = 0   
    train_acc_records = []   
    train_loss_records = [] 
    train_distance_records = []  
    val_acc_records = []   
    val_loss_records = []
    val_distance_records = []   
    for epoch in range(1, CONFIG["epoch"] + 1):
        train_acc, train_loss = train_epoch(
            train_loader, model, optimizer, time_criterion, appliance_criterion, epoch
        )  
        (
            val_acc,
            val_loss,
            day_real_targets,
            day_pred_targets,
            time_real_targets,
            time_pred_targets,
            appliance_real_targets,
            appliance_pred_targets,
        ) = evaluate(
            test_loader, model, scheduler, time_criterion, appliance_criterion, epoch
        )   

        train_acc_records.append(train_acc) 
        train_loss_records.append(train_loss)   
        val_acc_records.append(val_acc) 
        val_loss_records.append(val_loss)

        if epoch > CONFIG["min_epoch"]:
          if (val_loss < best_val_loss):   
              best_val_acc = val_acc   
              best_val_loss = val_loss
              torch.save(model.state_dict(), MODEL_PATH)
              time_lcs, time_edit_distance, time_hamming_distance = save_evaluate(
                  time_real_targets, time_pred_targets, TIME_LABELS, TIME_EVALUATE_PATH
              )
              appliance_lcs, appliance_edit_distance, appliance_hamming_distance = save_evaluate(
                  appliance_real_targets, appliance_pred_targets, APPLIANCE_LABELS, APPLIANCE_EVALUATE_PATH
              )
              #plot_confusion_matrix(
              #    time_real_targets, time_pred_targets, TIME_LABELS, TIME_CONFUSION_MATRIX_PATH
              #)
              plot_confusion_matrix(
                  appliance_real_targets, appliance_pred_targets, APPLIANCE_LABELS, APPLIANCE_CONFUSION_MATRIX_PATH
              )
              patience_counter = 0
              print("---------------------------------------------")
              print(f"edit_distance: {round((time_edit_distance + appliance_edit_distance)/2, 4)}")
              print(f"hamming_distance: {round((time_hamming_distance + appliance_hamming_distance)/2, 4)}")
              print(f"lcs: {round((time_lcs + appliance_lcs)/2, 4)}")
              print(f"best current val acc: {best_val_acc}")  
              print(f"best current val loss: {best_val_loss}")
              print("---------------------------------------------")
              print("\n")
          else:
              
              patience_counter += 1
              print(f"Patience: {patience_counter}")  
         
        if (patience_counter >= CONFIG["patience"] and epoch > CONFIG["min_epoch"]) or epoch >= CONFIG["epoch"]:
            print(f"best val acc: {best_val_acc}, training finished!")  
            break  

    pd.DataFrame(
        {
            "epoch": list(range(1, len(train_acc_records) + 1)),
            "train acc": train_acc_records,
            "val acc": val_acc_records,
        }
    ).to_csv(
        ACC_RECORDER_PATH, index=False
    )  
    epoch_visualization(train_acc_records, val_acc_records, "Acc", ACC_VISUALIZATION_PATH)  

    pd.DataFrame(
        {
            "epoch": list(range(1, len(train_loss_records) + 1)),
            "train loss": train_loss_records,
            "test loss": val_loss_records,
        }
    ).to_csv(
        LOSS_RECORDER_PATH, index=False
    )  
    
    epoch_visualization(train_loss_records, val_loss_records, "Loss", LOSS_VISUALIZATION_PATH)  
