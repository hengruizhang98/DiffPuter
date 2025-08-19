import sys
import pickle
import yaml
import os
import re
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
sys.path.append("..")
from data_utils import load_dataset, get_eval
from sklearn.preprocessing import MinMaxScaler

class uic_tabular_dataset(Dataset):
    def __init__(self, data, mask, use_index_list=None):
        
        # apply minmix scaler normalization
        self.minmax_scalar = MinMaxScaler()
        data_norm = self.minmax_scalar.fit_transform(data)

        self.eval_length = data.shape[1] # eval_length should be equal to attributes number.
        self.observed_values = data_norm
        self.observed_masks = ~np.isnan(data) 
        self.gt_masks = ~mask # False for missing values

        if use_index_list is None:
            self.use_index_list = np.arange(len(self.observed_values))
        else:
            self.use_index_list = use_index_list

    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        s = {
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "timepoints": np.arange(self.eval_length),
        }
        return s

    def __len__(self):
        return len(self.use_index_list)


