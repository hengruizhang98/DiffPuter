import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import os
import json

DATA_DIR = 'datasets'

def load_dataset(dataname, idx = 0, mask_type = 'MCAR', ratio = '30'):
    data_dir = f'datasets/{dataname}'
    info_path = f'datasets/Info/{dataname}.json'

    with open(info_path, 'r') as f:
        info = json.load(f)

    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    data_path = f'{data_dir}/data.csv'
    train_path = f'{data_dir}/train.csv'
    test_path = f'{data_dir}/test.csv'

    train_mask_path = f'{data_dir}/masks/rate{ratio}/{mask_type}/train_mask_{idx}.npy'
    test_mask_path = f'{data_dir}/masks/rate{ratio}/{mask_type}/test_mask_{idx}.npy'

    data_df = pd.read_csv(data_path)
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)


    train_mask = np.load(train_mask_path)
    test_mask = np.load(test_mask_path)

    cols = train_df.columns

    data_num = data_df[cols[num_col_idx]].values.astype(np.float32)
    data_cat = data_df[cols[cat_col_idx]].astype(str)
    data_y = data_df[cols[target_col_idx]]

    train_num = train_df[cols[num_col_idx]].values.astype(np.float32)
    train_cat = train_df[cols[cat_col_idx]].astype(str)
    train_y = train_df[cols[target_col_idx]]


    test_num = test_df[cols[num_col_idx]].values.astype(np.float32)
    test_cat = test_df[cols[cat_col_idx]].astype(str)
    test_y = test_df[cols[target_col_idx]]
    
    cat_columns = data_cat.columns

    train_cat_idx, test_cat_idx = None, None
    extend_train_mask = None
    extend_test_mask = None
    cat_bin_num = None


    # only contain numerical features

    if len(cat_col_idx) == 0:
        train_X = train_num
        test_X = test_num

        extend_train_mask = train_mask[:, num_col_idx]
        extend_test_mask = test_mask[:, num_col_idx]

    # Contain both numerical and categorical features

    else:

        if not os.path.exists(f'{data_dir}/{cat_columns[0]}_map.json'):

            for column in cat_columns:
                map_path_bin = f'{data_dir}/{column}_map_bin.json'
                map_path_idx = f'{data_dir}/{column}_map_idx.json'
                categories = data_cat[column].unique()
                num_categories = len(categories) 

                num_bits = (num_categories - 1).bit_length()

                category_to_binary = {category: format(index, '0' + str(num_bits) + 'b') for index, category in enumerate(categories)}
                category_to_idx = {category: index for index, category in enumerate(categories)}

            #     for category, binary_encoding in category_to_binary.items():
            #         print(f'{category}: {binary_encoding}')

                with open(map_path_bin, 'w') as f:
                    json.dump(category_to_binary, f)
                with open(map_path_idx, 'w') as f:
                    json.dump(category_to_idx, f)

        train_cat_bin = []
        test_cat_bin = []

        train_cat_idx = []
        test_cat_idx = []
        cat_bin_num = []
                
        for column in cat_columns:
            map_path_bin = f'{data_dir}/{column}_map_bin.json'
            map_path_idx = f'{data_dir}/{column}_map_idx.json'
            
            with open(map_path_bin, 'r') as f:
                category_to_binary = json.load(f)
            with open(map_path_idx, 'r') as f:
                category_to_idx = json.load(f)
                
            train_cat_enc_i = train_cat[column].map(category_to_binary).to_numpy()
            train_cat_idx_i = train_cat[column].map(category_to_idx).to_numpy().astype(np.int64)
            train_cat_bin_i = np.array([list(map(int, binary)) for binary in train_cat_enc_i])

            test_cat_enc_i = test_cat[column].map(category_to_binary).to_numpy()
            test_cat_idx_i = test_cat[column].map(category_to_idx).to_numpy().astype(np.int64)
            test_cat_bin_i = np.array([list(map(int, binary)) for binary in test_cat_enc_i])

            train_cat_bin.append(train_cat_bin_i)
            test_cat_bin.append(test_cat_bin_i)
            
            train_cat_idx.append(train_cat_idx_i)
            test_cat_idx.append(test_cat_idx_i)
            cat_bin_num.append(train_cat_bin_i.shape[1])
                
        train_cat_bin = np.concatenate(train_cat_bin, axis = 1).astype(np.float32)
        test_cat_bin = np.concatenate(test_cat_bin, axis = 1).astype(np.float32)

        train_cat_idx = np.stack(train_cat_idx, axis = 1)
        test_cat_idx = np.stack(test_cat_idx, axis = 1)

        cat_bin_num = np.array(cat_bin_num)

        train_X = np.concatenate([train_num, train_cat_bin], axis = 1)
        test_X = np.concatenate([test_num, test_cat_bin], axis = 1)

        train_num_mask = train_mask[:, num_col_idx]
        train_cat_mask = train_mask[:, cat_col_idx]
        test_num_mask = test_mask[:, num_col_idx]
        test_cat_mask = test_mask[:, cat_col_idx]

        def extend_mask(mask, bin_num):

            num_rows, num_cols = mask.shape
            cum_sum = bin_num.cumsum()
            cum_sum = np.insert(cum_sum, 0, 0)
            result = np.zeros((num_rows, bin_num.sum() ), dtype=bool)
            
            for idx in range(num_cols):
                res = np.tile(mask[:, idx][:, np.newaxis], bin_num[idx])
                result[:, cum_sum[idx]:cum_sum[idx + 1]] = res
                
            return result

        train_cat_mask = extend_mask(train_cat_mask, cat_bin_num)
        test_cat_mask = extend_mask(test_cat_mask, cat_bin_num)

        extend_train_mask = np.concatenate([train_num_mask, train_cat_mask], axis = 1)
        extend_test_mask = np.concatenate([test_num_mask, test_cat_mask], axis = 1)

    return train_X, test_X, train_mask, test_mask, train_num, test_num, train_cat_idx, test_cat_idx, extend_train_mask, extend_test_mask, cat_bin_num

def mean_std(data, mask):
    mask = ~mask
    mask = mask.astype(np.float32)
    mask_sum = mask.sum(0)
    mask_sum[mask_sum == 0] = 1
    mean = (data * mask).sum(0) / mask_sum
    var = ((data - mean) ** 2 * mask).sum(0) / mask_sum
    std = np.sqrt(var)
    return mean, std

def get_eval(dataname, X_recon, X_true, truth_cat_idx, num_num, cat_bin_num, mask, oos = False):

    data_dir = f'datasets/{dataname}'

    info_path = f'datasets/Info/{dataname}.json'

    with open(info_path, 'r') as f:
        info = json.load(f)
    
    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']

    num_mask = mask[:, num_col_idx].astype(bool)
    cat_mask = mask[:, cat_col_idx].astype(bool)

    num_pred = X_recon[:, :num_num]
    cat_pred = X_recon[:, num_num:]

    num_cat = len(cat_col_idx)

    num_true = X_true[:, :num_num]
    cat_true = truth_cat_idx

    if dataname == 'news' and oos == True:
        num_mask = np.delete(num_mask, 6265, axis=0)
        num_pred = np.delete(num_pred, 6265, axis=0)
        num_true = np.delete(num_true, 6265, axis=0)

    div = num_pred[num_mask] - num_true[num_mask]
    mae = np.abs(div).mean()
    rmse = np.sqrt((div**2).mean())
        
    mae = np.abs(div).mean()
    rmse = np.sqrt((div**2).mean())

    return mae, rmse
