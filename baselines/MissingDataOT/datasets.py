import numpy as np
import pandas as pd
from torch.utils.data import Dataset

DATA_DIR = 'datasets'


class pair_dataset(Dataset):
    def __init__(self, X, X_miss):
        self.X = X 
        self.X_miss = X_miss
    
    def __getitem__(self, idx):
        return self.X[idx], self.X_miss[idx]

    def __len__(self):
        return self.X.shape[0]


def load_dataset(name):
    data_dir = f'{DATA_DIR}/{name}'

    train_path = f'{data_dir}/train.csv'
    test_path = f'{data_dir}/test.csv'

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_X = train_df.values[:, :-1].astype(np.float32)
    train_y =  train_df.values[:, -1]

    test_X = test_df.values[:, :-1].astype(np.float32)
    test_y =  test_df.values[:, -1]

    return train_X, test_X