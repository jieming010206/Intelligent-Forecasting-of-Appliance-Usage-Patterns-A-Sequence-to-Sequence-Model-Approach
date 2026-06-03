import os
import random

import numpy as np
import torch


def makedir(_dir):
    # make dictionary
    os.makedirs(_dir, exist_ok=True)


def setup_seed(seed=42):
    # set random seed ensure the result is reproducible
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    torch.use_deterministic_algorithms(True)


# Chose model
MODEL_NAME = "5e-6_lstm_0.3"  

# Input sequence length
INPUT_SEQ_LEN = 60

# Output sequence length
OUTPUT_SEQ_LEN = 12

# All labels
# LABELS = [f"E{i}" for i in range(0, 12)]
DAY_LABELS = [f"D{i}" for i in range(7)]
TIME_LABELS = [f"E{i}" for i in range(144)]
APPLIANCE_LABELS = ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10"]

# oot dictionary
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

# Data
DATA_DIR = os.path.join(ROOT_DIR, "data")
DATA_PATH = os.path.join(DATA_DIR, "household_0.csv")

# Model
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs", MODEL_NAME)

MODEL_PATH = os.path.join(OUTPUTS_DIR, "model.pkl")

ACC_RECORDER_PATH = os.path.join(OUTPUTS_DIR, "acc_recorder.csv")
ACC_VISUALIZATION_PATH = os.path.join(OUTPUTS_DIR, "acc_visualization.png")
LOSS_RECORDER_PATH = os.path.join(OUTPUTS_DIR, "loss_recorder.csv")
LOSS_VISUALIZATION_PATH = os.path.join(OUTPUTS_DIR, "loss_visualization.png")

TIME_EVALUATE_PATH = os.path.join(OUTPUTS_DIR, "time_evaluate.txt")
APPLIANCE_EVALUATE_PATH = os.path.join(OUTPUTS_DIR, "appliance_evaluate.txt")
DURATION_EVALUATE_PATH = os.path.join(OUTPUTS_DIR, "duration_evaluate.txt")

TIME_CONFUSION_MATRIX_PATH = os.path.join(OUTPUTS_DIR, "time_confusion_matrix.png")
APPLIANCE_CONFUSION_MATRIX_PATH = os.path.join(OUTPUTS_DIR, "appliance_confusion_matrix.png")
DURATION_CONFUSION_MATRIX_PATH = os.path.join(OUTPUTS_DIR, "duration_confusion_matrix.png")

# Make dictionary
makedir(OUTPUTS_DIR)

# Set random seed, Ensure the result is reproduceble
setup_seed(seed=42)

# Choice device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: %s" % DEVICE)

# parameter
CONFIG = {
    "batch_size": 32,
    "lr": 5e-6,
    "epoch": 500,
    "min_epoch": 50,
    "patience": 20,
}
