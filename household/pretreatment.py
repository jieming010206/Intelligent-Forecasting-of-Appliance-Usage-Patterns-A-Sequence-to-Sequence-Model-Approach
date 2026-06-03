from collections import Counter

import numpy as np
import pandas as pd

from configs import INPUT_SEQ_LEN, OUTPUT_SEQ_LEN, DATA_PATH


def load_data(input_seq_len, output_seq_len, k):
    # Load data
    data = pd.read_csv("data/new_data_0.csv")
    X_train, y_train, X_val, y_val = [], [], [], []

    total = len(data)
    split = total // 5
    valid_start = split * (k-1)
    valid_end = split * k
    for i in range(input_seq_len + output_seq_len, total):
        start = i - input_seq_len - output_seq_len
        end = i
        
        if start > valid_start and end < valid_end: 
            X_val.append(data.iloc[i - input_seq_len - output_seq_len : i - output_seq_len, 2:].values)  
            y_val.append(data.iloc[i - output_seq_len : i, 3:].values) 
        if end < valid_start or start > valid_end:  
            X_train.append(data.iloc[i - input_seq_len - output_seq_len : i - output_seq_len, 2:].values)  
            y_train.append(data.iloc[i - output_seq_len : i, 3:].values) 

    return (
        np.array(X_train).tolist(),
        np.array(X_val).tolist(),
        np.array(y_train).tolist(),
        np.array(y_val).tolist(),
    )
    
def load_data_2(input_seq_len, output_seq_len):
    # Load data
    data = pd.read_csv("data/house_ids_0.csv")
    X_train, y_train, X_val, y_val = [], [], [], []

    total = len(data)
    for i in range(input_seq_len + output_seq_len, total):
        start = data.iloc[i - input_seq_len - output_seq_len, 0]
        end = data.iloc[i, 0]
        
        if (start % 5 == 4) and (end % 5 == 4): 
            X_val.append(data.iloc[i - input_seq_len - output_seq_len : i - output_seq_len, 2:].values)  
            y_val.append(data.iloc[i - output_seq_len : i, -2:].values) 
        if (start % 5 < 4) and (end % 5 < 4):  
            X_train.append(data.iloc[i - input_seq_len - output_seq_len : i - output_seq_len, 2:].values)  
            y_train.append(data.iloc[i - output_seq_len : i, -2:].values) 

    return (
        np.array(X_train).tolist(),
        np.array(X_val).tolist(),
        np.array(y_train).tolist(),
        np.array(y_val).tolist(),
    )

    
def load_merge_data(input_seq_len, output_seq_len, num = 2, k = 1):
    # load data
    X_train, Y_train, X_val, Y_val = [], [], [], []
    for i in range(num):
        path = f"data/data_{i}.csv"
        data = pd.read_csv(path)  # read data from path

        x_train, y_train, x_val, y_val = [], [], [], []

        total = len(data)
        split = total // 5
        valid_start = split * (k-1)
        valid_end = split * k
        for i in range(input_seq_len + output_seq_len, total):
            start = i - input_seq_len - output_seq_len
            end = i
            
            if start > valid_start and end < valid_end: 
                x_val.append(data.iloc[i - input_seq_len - output_seq_len : i - output_seq_len, 2:].values)  
                y_val.append(data.iloc[i - output_seq_len : i, -2:].values) 
            if end < valid_start or start > valid_end:  
                x_train.append(data.iloc[i - input_seq_len - output_seq_len : i - output_seq_len, 2:].values)  
                y_train.append(data.iloc[i - output_seq_len : i, -2:].values)  
        
        X_train.append(x_train)
        Y_train.append(y_train)
        X_val.append(x_val)
        Y_val.append(y_val)

    
    X_train = np.concatenate(X_train)
    Y_train = np.concatenate(Y_train)
    X_val = np.concatenate(X_val)
    Y_val = np.concatenate(Y_val)
    
    return X_train, X_val, Y_train, Y_val

def load_merge_data_2(input_seq_len, output_seq_len):
    # Load data
    X_train, Y_train, X_val, Y_val = [], [], [], []
    for i in range(8):
        x_train, y_train, x_val, y_val = [], [], [], []
        path = f"data/house_ids_{i}.csv"
        data = pd.read_csv(path)  # read data from path

        total = len(data)
        for i in range(input_seq_len + output_seq_len, total):
            start = data.iloc[i - input_seq_len - output_seq_len, 0]
            end = data.iloc[i, 0]
            
            if (start % 5 == 4) and (end % 5 == 4): 
                x_val.append(data.iloc[i - input_seq_len - output_seq_len : i - output_seq_len, 2:].values)  
                y_val.append(data.iloc[i - output_seq_len : i, -2:].values) 
            if (start % 5 < 4) and (end % 5 < 4):  
                x_train.append(data.iloc[i - input_seq_len - output_seq_len : i - output_seq_len, 2:].values)  
                y_train.append(data.iloc[i - output_seq_len : i, -2:].values) 
        if len(x_train) == 0 or len(x_val) == 0:
            continue
        X_train.append(x_train)
        Y_train.append(y_train)
        X_val.append(x_val)
        Y_val.append(y_val)

    X_train = np.concatenate(X_train)
    Y_train = np.concatenate(Y_train)
    X_val = np.concatenate(X_val)
    Y_val = np.concatenate(Y_val)
    
    return X_train, X_val, Y_train, Y_val


if __name__ == "__main__":
    X_train, X_val, Y_train, Y_val = load_merge_data_2(INPUT_SEQ_LEN, OUTPUT_SEQ_LEN)
    a =  np.array(Y_train)[:, :, 0].flatten().tolist()
    print(np.shape(X_train))
    print(np.shape(X_val))
    print(np.shape(Y_train))
    print(np.shape(Y_val))
    print(len(np.unique(a)))
