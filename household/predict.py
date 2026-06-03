import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from datasets import Datasets
from pretreatment import load_data_2
from train import train


from configs import (
    MODEL_NAME,
    INPUT_SEQ_LEN,
    OUTPUT_SEQ_LEN,
    TIME_LABELS,
    APPLIANCE_LABELS,
    MODEL_PATH,
    DEVICE,
    CONFIG,
)
from models.lstm import Model
MODEL_PATH = "outputs/lstm_0.7/model.pkl"

model = Model(
    len(TIME_LABELS),
    len(APPLIANCE_LABELS),
    OUTPUT_SEQ_LEN,
)  # define model structure
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))  # load model 
model = model.to(DEVICE)  # use CPU/GPU
model.eval()  # model evaluation

if __name__ == "__main__":
    import random
    indices = [i for i in range(32)]
    print(indices)
    # Assuming load_data function loads the data properly
    X_train, X_val, y_train, y_val = load_data_2(INPUT_SEQ_LEN, OUTPUT_SEQ_LEN)
    test_datasets = Datasets(X_val, y_val)  # Assuming Datasets class is defined properly
        
    test_loader = DataLoader(
        test_datasets,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )  # training dataloader

    # Get the first batch
    first_batch = next(iter(test_loader))
    inputs, targets = first_batch  # Assuming that the train_loader returns inputs and targets

    # Send the inputs to the device
    inputs = torch.LongTensor(inputs).to(DEVICE)

    # Forward pass through the model
    time_outputs, appliance_outputs = model(inputs)

    # Get the predictions
    time_outputs = torch.argmax(time_outputs, dim=2).cpu().numpy()
    appliance_outputs= torch.argmax(appliance_outputs, dim=2).cpu().numpy()

    # Assuming targets are shaped properly
    time_targets = targets[:,:,0].cpu().numpy()
    appliance_targets = targets[:, :, 1].cpu().numpy()

    
    print("Model:", MODEL_NAME)
    for index in indices:  # Printing the first four sequences
        print(f"Sample: {index}")
        print("target")
        tagret_dataframe = pd.DataFrame({"time":time_targets[index], "appliance": appliance_targets[index]})
        print(tagret_dataframe)
        print("prediction")
        prediction_dataframe = pd.DataFrame({"time":time_outputs[index], "appliance": appliance_outputs[index]})
        print(prediction_dataframe)
