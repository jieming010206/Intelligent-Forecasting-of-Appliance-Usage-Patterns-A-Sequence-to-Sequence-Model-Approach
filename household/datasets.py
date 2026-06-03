import torch
from torch.utils.data import Dataset

from configs import INPUT_SEQ_LEN, OUTPUT_SEQ_LEN
from pretreatment import load_data


class Datasets(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        assert len(self.X) == len(self.y)
        return len(self.y)

    def __getitem__(self, idx):
        # return torch.LongTensor(self.X[idx]), torch.LongTensor([item[0] for item in self.y[idx]][:1])
        return torch.LongTensor(self.X[idx]), torch.LongTensor(self.y[idx])


if __name__ == "__main__":
    X_train, X_val, y_train, y_val = load_data(INPUT_SEQ_LEN, OUTPUT_SEQ_LEN)  # load data

    train_datasets = Datasets(X_train, y_train)  # training set
    val_datasets = Datasets(X_val, y_val)  # validation set

    print(f"train data length: {len(train_datasets)}, val data length: {len(val_datasets)}")  

    for idx, (features, targets) in enumerate(train_datasets): 
        print(features.shape, targets.shape)  # dimension
        print(features, targets)  # examples
        break
