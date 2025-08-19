import sys
import numpy as np
import pandas as pd
from hyperimpute.plugins.imputers import Imputers
sys.path.append("..")
from data_utils import load_dataset, get_eval, get_subset_idx
import argparse
import json
import os
import pdb 

methods = [
 'median',
 'softimpute',
 'gain',
 'sklearn_missforest',
 'most_frequent',
 'mice',
 'EM',
 'miracle',
 'ice',
 'mean',
 'miwae',
 'hyperimpute',
 ]

parser = argparse.ArgumentParser(description='Missing Value Imputation')

#parser.add_argument('--dataname', type=str, default='california', help='Name of dataset.')
parser.add_argument('--mask_type', type=str, default='MCAR')
parser.add_argument('-retrain', action='store_true', help='Retrain the models or not')
parser.add_argument('--mask_num', type=int, default=10) 
parser.add_argument('--missing_rate', type=float, default=0.3)
parser.add_argument('-benchmark_sample_size', action='store_true', help='Benchmark sample size or not')
parser.add_argument('-benchmark_mask_ratio', action='store_true', help='Benchmark mask ratio or not')


args = parser.parse_args()

datanames = ['bean', 'california', 'adult', 'beijing', 'default', 'gesture', 'letter', 'magic', 'news', 'shoppers'] 

mask_type = args.mask_type
mask_num = args.mask_num
retrain = args.retrain
benchmark_sample_size = args.benchmark_sample_size
benchmark_mask_ratio = args.benchmark_mask_ratio

if benchmark_sample_size:
    subset_idx = get_subset_idx(total_num=29229, target_num=[1000, 6000, 11000, 16000]) #target_num need to be increase order
    methods = ['hyperimpute', 'EM', 'sklearn_missforest']
    mask_num = 3

for method in methods:
    if benchmark_mask_ratio:
        missing_rates = [0.5, 0.7]
    else:
        missing_rates = [args.missing_rate] #0.3

    if benchmark_sample_size:
        assert args.mask_type == 'MCAR', 'Benchmark sample size only supports MCAR mask.'
    
    if benchmark_mask_ratio:
        datanames = ['news', 'adult', 'beijing', 'default']
    elif benchmark_sample_size:
        datanames = ['beijing']
    else:
        datanames = ['bean', 'adult', 'beijing', 'california', 'default', 'gesture', 'letter', 'magic', 'news', 'shoppers']
        
    for dataname in datanames:
        for missing_rate in missing_rates:

            print('dataname:', dataname)
            
            DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'datasets'))

            info_path = f'{DATA_DIR}/Info/{dataname}.json'

            if missing_rate == 0.3:
                folder_name = 'rate30'
            elif missing_rate == 0.5:
                folder_name = 'rate50'
            elif missing_rate == 0.7:
                folder_name = 'rate70'
            else:
                raise ValueError('Invalid missing rate, please choose from 0.3, 0.5, 0.7.')
            
            if mask_type == 'MCAR':
                mask_dir = f'{DATA_DIR}/{dataname}/masks/{folder_name}/MCAR/' 
            elif mask_type == 'MAR':
                mask_dir = f'{DATA_DIR}/{dataname}/masks/{folder_name}/MAR/'
            elif mask_type == 'MNAR':
                mask_dir = f'{DATA_DIR}/{dataname}/masks/{folder_name}/MNAR_logistic_T2/'
            else:
                raise ValueError('Invalid mask type, please choose from MCAR, MAR, MNAR.')

            for mask_idx in range(mask_num):   

                for sample_size in [1000, 6000, 11000, 16000, 'all']:

                    if benchmark_sample_size:
                        if sample_size == 'all':
                            continue
                    else:
                        if sample_size != 'all':
                            continue

                    train_X, test_X, ori_train_mask, ori_test_mask, train_num, test_num, train_cat_idx, test_cat_idx, train_mask, test_mask,cat_bin_num = load_dataset(dataname, mask_idx, mask_dir=mask_dir)
        
                    if benchmark_sample_size:
                        row_idx = subset_idx[sample_size]
                    else:
                        row_idx = np.arange(train_X.shape[0])

                    train_X, ori_train_mask, train_num, train_mask = train_X[row_idx], ori_train_mask[row_idx], train_num[row_idx], train_mask[row_idx]
                    if train_cat_idx is not None:
                        train_cat_idx = train_cat_idx[row_idx] 

                    print('mask_idx:', mask_idx, 'method:', method, 'dataname:', dataname)
                    
                    if not args.benchmark_sample_size and not args.benchmark_mask_ratio:
                        ckpt_train = f'stat_baseline_output/{mask_type}/filled_X/{dataname}/mask_{mask_idx}/train/'
                        os.makedirs(ckpt_train) if not os.path.exists(ckpt_train) else None

                        ckpt_in_sample = f'stat_baseline_output/{mask_type}/in_sample_result/{dataname}/'
                        os.makedirs(ckpt_in_sample) if not os.path.exists(ckpt_in_sample) else None
                    elif args.benchmark_sample_size:
                        ckpt_train = f'stat_baseline_output/sensitive_analysis/sample_ratio/sample_size{sample_size}/{mask_type}/filled_X/{dataname}/mask_{mask_idx}/train/'
                        os.makedirs(ckpt_train) if not os.path.exists(ckpt_train) else None

                        ckpt_in_sample = f'stat_baseline_output/sensitive_analysis/sample_ratio/sample_size{sample_size}/{mask_type}/in_sample_result/{dataname}/'
                        os.makedirs(ckpt_in_sample) if not os.path.exists(ckpt_in_sample) else None
                    elif args.benchmark_mask_ratio:
                        ckpt_train = f'stat_baselin[[e_output/sensitive_analysis/mask_ratio/{folder_name}/{mask_type}/filled_X/{dataname}/mask_{mask_idx}/train/'
                        os.makedirs(ckpt_train) if not os.path.exists(ckpt_train) else None

                        ckpt_in_sample = f'stat_baseline_output/sensitive_analysis/mask_ratio/{folder_name}/{mask_type}/in_sample_result/{dataname}/'
                        os.makedirs(ckpt_in_sample) if not os.path.exists(ckpt_in_sample) else None

                    if os.path.exists(f'{ckpt_train}/{method}.npy'):
                        print(f'{method} already exists, skipping...')
                        continue

                    # normalize data
                    mean_train_X, mean_test_X = train_X.mean(0), test_X.mean(0)
                    std_train_X, std_test_X = train_X.std(0), test_X.std(0)
                    # avoid division by zero
                    std_train_X, std_test_X = np.where(std_train_X == 0, 1e-3, std_train_X), np.where(std_test_X == 0, 1e-3, std_test_X)
                    train_X = (train_X - mean_train_X) / std_train_X 
                    test_X = (test_X - mean_train_X) / std_train_X

                    # apply mask
                    x_train_miss = np.copy(train_X)
                    x_train_miss[train_mask] = np.nan

                    X_train_true_np = np.copy(train_X)
                
                    X = pd.DataFrame(x_train_miss)

                    imputer = Imputers().get(method)

                    X_filled = imputer.fit_transform(X.copy())
                    X_filled = X_filled.to_numpy()

                    pred_X = X_filled[:]
                    len_num = train_num.shape[1]
                    res = pred_X[:, len_num:] * std_train_X[len_num:] + mean_train_X[len_num:]
                    pred_X[:, len_num:] = res

                    mae, rmse, acc = 0.0,0.0,0.0
                    
                    mae, rmse, acc = get_eval(dataname, pred_X, X_train_true_np, train_cat_idx, train_num.shape[1], cat_bin_num, ori_train_mask)

                    print(f'Saving {method} results...')  
                    # 1. save imputation   
                    np.save(f'{ckpt_train}/{method}.npy', X_filled) 
                    # 2. write in_sample performance
                    with open(f'{ckpt_in_sample}/in_sample_result_mask{mask_idx}.txt', 'a+') as file: 
                        file.write(f"{method} in sample MAE: {mae}\n")
                        file.write(f"{method} in sample RMSE: {rmse}\n")
                        file.write(f"{method} in sample acc: {acc}\n")

        