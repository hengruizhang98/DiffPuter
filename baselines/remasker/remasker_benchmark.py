import sys
import numpy as np
import pandas as pd
import torch
import argparse
import json
import os
import pdb 

from remasker_impute import ReMasker
from utils import get_args_parser
sys.path.append("..")
from data_utils import load_dataset, get_eval, get_subset_idx

args = get_args_parser().parse_args()

if args.benchmark_mask_ratio:
    #datanames = ['news', 'adult', 'beijing', 'default']
    datanames = ['beijing']
elif args.benchmark_sample_size:
    datanames = ['beijing']
else:
    datanames = ['bean', 'adult', 'beijing', 'california', 'default', 'gesture', 'letter', 'magic', 'news', 'shoppers']
    
if args.benchmark_mask_ratio:
    missing_rates = [0.5, 0.7]
else:
    missing_rates = [args.missing_rate] #0.3

if args.benchmark_sample_size:
    assert args.mask_type == 'MCAR', 'Benchmark sample size only supports MCAR mask.'

mask_type = args.mask_type
mask_num = args.mask_num
retrain = args.retrain
#missing_rate = args.missing_rate
benchmark_sample_size = args.benchmark_sample_size
benchmark_mask_ratio = args.benchmark_mask_ratio
if benchmark_sample_size:
    subset_idx = get_subset_idx(total_num=29229, target_num=[1000, 6000, 11000, 16000]) #target_num need to be increase order


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

                if not args.benchmark_sample_size and not args.benchmark_mask_ratio:
                    ckpt_train = f'Remasker_output/{mask_type}/filled_X/{dataname}/mask_{mask_idx}/train/'
                    os.makedirs(ckpt_train) if not os.path.exists(ckpt_train) else None

                    ckpt_test = f'Remasker_output/{mask_type}/filled_X/{dataname}/mask_{mask_idx}/test/'
                    os.makedirs(ckpt_test) if not os.path.exists(ckpt_test) else None

                    ckpt_in_sample = f'Remasker_output/{mask_type}/in_sample_result/{dataname}/'
                    os.makedirs(ckpt_in_sample) if not os.path.exists(ckpt_in_sample) else None

                    ckpt_out_sample = f'Remasker_output/{mask_type}/out_sample_result/{dataname}/'
                    os.makedirs(ckpt_out_sample) if not os.path.exists(ckpt_out_sample) else None
                
                elif args.benchmark_sample_size:
                    ckpt_train = f'Remasker_output/sensitive_analysis/sample_ratio/sample_size{sample_size}/{mask_type}/filled_X/{dataname}/mask_{mask_idx}/train/'
                    os.makedirs(ckpt_train) if not os.path.exists(ckpt_train) else None

                    ckpt_test = f'Remasker_output/sensitive_analysis/sample_ratio/sample_size{sample_size}/{mask_type}/filled_X/{dataname}/mask_{mask_idx}/test/'
                    os.makedirs(ckpt_test) if not os.path.exists(ckpt_test) else None

                    ckpt_in_sample = f'Remasker_output/sensitive_analysis/sample_ratio/sample_size{sample_size}/{mask_type}/in_sample_result/{dataname}/'
                    os.makedirs(ckpt_in_sample) if not os.path.exists(ckpt_in_sample) else None

                    ckpt_out_sample = f'Remasker_output/sensitive_analysis/sample_ratio/sample_size{sample_size}/{mask_type}/out_sample_result/{dataname}/'
                    os.makedirs(ckpt_out_sample) if not os.path.exists(ckpt_out_sample) else None
                
                elif args.benchmark_mask_ratio:
                    ckpt_train = f'Remasker_output/sensitive_analysis/mask_ratio/{folder_name}/{mask_type}/filled_X/{dataname}/mask_{mask_idx}/train/'
                    os.makedirs(ckpt_train) if not os.path.exists(ckpt_train) else None

                    ckpt_test = f'Remasker_output/sensitive_analysis/mask_ratio/{folder_name}/{mask_type}/filled_X/{dataname}/mask_{mask_idx}/test/'
                    os.makedirs(ckpt_test) if not os.path.exists(ckpt_test) else None

                    ckpt_in_sample = f'Remasker_output/sensitive_analysis/mask_ratio/{folder_name}/{mask_type}/in_sample_result/{dataname}/'
                    os.makedirs(ckpt_in_sample) if not os.path.exists(ckpt_in_sample) else None

                    ckpt_out_sample = f'Remasker_output/sensitive_analysis/mask_ratio/{folder_name}/{mask_type}/out_sample_result/{dataname}/'
                    os.makedirs(ckpt_out_sample) if not os.path.exists(ckpt_out_sample) else None
                
                if os.path.exists(f'{ckpt_test}/remasker.npy'):
                    print(f'Remasker at {ckpt_test}/remasker.npy already exists, skipping...')
                    continue

                # apply mask
                x_train_miss = np.copy(train_X)
                x_train_miss[train_mask] = np.nan

                x_test_miss = np.copy(test_X)
                x_test_miss[test_mask] = np.nan

                X_train_true_np = np.copy(train_X)
                X_test_true_np = np.copy(test_X)

                # normalize gt for evaluation
                mean_train_X = X_train_true_np.mean(0)
                std_train_X = X_train_true_np.std(0)
                # avoid division by zero
                std_train_X = np.where(std_train_X == 0, 1e-3, std_train_X)
                X_train_true_np = (X_train_true_np - mean_train_X) / std_train_X 
                X_test_true_np = (X_test_true_np - mean_train_X) / std_train_X

                X = pd.DataFrame(x_train_miss)
                X_test = pd.DataFrame(x_test_miss)

                imputer = ReMasker()
                X_filled = imputer.fit_transform(X)
                X_filled_test = imputer.transform(torch.tensor(X_test.values, dtype=torch.float32)).numpy()

                # normalize X_filled for evaluation
                mean_train_X = X_filled.mean(0)
                std_train_X = X_filled.std(0)
                # avoid division by zero
                std_train_X = np.where(std_train_X == 0, 1e-3, std_train_X)
                X_filled = (X_filled - mean_train_X) / std_train_X

                # normalize X_filled_test for evaluation
                mean_train_X = X_filled_test.mean(0)
                std_train_X = X_filled_test.std(0)
                # avoid division by zero
                std_train_X = np.where(std_train_X == 0, 1e-3, std_train_X)
                X_filled_test = (X_filled_test - mean_train_X) / std_train_X
            
                pred_X = X_filled[:]
                len_num = train_num.shape[1]
                res = pred_X[:, len_num:] * std_train_X[len_num:] + mean_train_X[len_num:]
                pred_X[:, len_num:] = res  #Equivalent to only normalize the numerical columns

                pred_X_test = X_filled_test[:]
                len_num = train_num.shape[1]
                res = pred_X_test[:, len_num:] * std_train_X[len_num:] + mean_train_X[len_num:]
                pred_X_test[:, len_num:] = res
                        
                mae_tr, rmse_tr, acc_tr = 0.0,0.0,0.0
                mae_te, rmse_te, acc_te = 0.0,0.0,0.0
                
                mae_tr, rmse_tr, acc_tr = get_eval(dataname, pred_X, X_train_true_np, train_cat_idx, train_num.shape[1], cat_bin_num, ori_train_mask)
                mae_te, rmse_te, acc_te = get_eval(dataname, pred_X_test, X_test_true_np, test_cat_idx, test_num.shape[1], cat_bin_num, ori_test_mask)

                print('Saving Remasker results...')  
                # 1. save imputation   
                np.save(f'{ckpt_train}/remasker.npy', X_filled) 
                np.save(f'{ckpt_test}/remasker.npy', X_filled_test)

                # 2. write in_sample performance
                with open(f'{ckpt_in_sample}/in_sample_result_mask{mask_idx}.txt', 'a+') as file: 
                    file.write(f"Remasker in sample MAE: {mae_tr}\n")
                    file.write(f"Remasker in sample RMSE: {rmse_tr}\n")
                    file.write(f"Remasker in sample acc: {acc_tr}\n")
                
                # 3. write out_sample performance
                with open(f'{ckpt_out_sample}/out_sample_result_mask{mask_idx}.txt', 'a+') as file: 
                    file.write(f"Remasker out sample MAE: {mae_te}\n")
                    file.write(f"Remasker out sample RMSE: {rmse_te}\n")
                    file.write(f"Remasker out sample acc: {acc_te}\n")
            

