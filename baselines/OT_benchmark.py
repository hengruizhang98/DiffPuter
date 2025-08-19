import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import numpy as np

from sklearn.preprocessing import scale
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer

import os

#from geomloss import SamplesLoss

from MissingDataOT import nanmean, pick_epsilon, MAE, RMSE, dataset_loader, softimpute, cv_softimpute
from MissingDataOT import OTimputer, RRimputer

from TDM import run_TDM

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.debug("test")

from data_utils import load_dataset, get_eval, get_subset_idx

import argparse
import json

import pdb

torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='Missing Value Imputation')

#parser.add_argument('--dataname', type=str, default='california', help='Name of dataset.')
parser.add_argument('--niter', type=int, default=12000)
parser.add_argument('--max_iter', type=int, default=15)
parser.add_argument('--niter_rr', type=int, default=15)
parser.add_argument('--mask_num', type=int, default=10) 
parser.add_argument('--mask_type', type=str, default='MCAR')
parser.add_argument('--missing_rate', type=float, default=0.3)
parser.add_argument('-retrain', action='store_true', help='Retrain the models or not')

parser.add_argument('-benchmark_sample_size', action='store_true', help='Benchmark sample size or not')
parser.add_argument('-benchmark_mask_ratio', action='store_true', help='Benchmark mask ratio or not')
#parser.add_argument('-re_eval', action='store_true', help='Re-evaluate the models or not')

args = parser.parse_args()

if __name__ == '__main__':
    #dataname = args.dataname 
    retrain = args.retrain
    mask_type = args.mask_type
    N_ITER = args.niter #12000 
    MAX_ITER = args.max_iter #15
    NITER_RR = args.niter_rr #15
    mask_num = args.mask_num 
    benchmark_sample_size = args.benchmark_sample_size
    benchmark_mask_ratio = args.benchmark_mask_ratio
    if benchmark_sample_size:
        subset_idx = get_subset_idx(total_num=29229, target_num=[1000, 6000, 11000, 16000]) #target_num need to be increase order

    if benchmark_mask_ratio:
        missing_rates = [0.5, 0.7]
    else:
        missing_rates = [args.missing_rate] #0.3

    if benchmark_sample_size:
        assert mask_type == 'MCAR', 'Benchmark sample size only supports MCAR mask.'
    
    if benchmark_mask_ratio:
        #datanames = ['news', 'adult', 'beijing', 'default']
        datanames = ['beijing']
    elif benchmark_sample_size:
        datanames = ['beijing']
    else:
        datanames = ['bean', 'adult', 'beijing', 'california', 'default', 'gesture', 'letter', 'magic', 'news', 'shoppers'] 
        
    for dataname in datanames:
        for missing_rate in missing_rates:
            print('dataname:', dataname)
            
            DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets'))

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

            with open(info_path, 'r') as f:
                info = json.load(f)

            cat_col_idx = info['cat_col_idx']
            if len(cat_col_idx) == 0:
                is_cat_dataset = False
            else:
                is_cat_dataset = True

            for mask_idx in range(mask_num):
                
                for sample_size in [1000, 6000, 11000, 16000, 'all']:

                    if benchmark_sample_size:
                        if sample_size == 'all':
                            continue
                    else:
                        if sample_size != 'all':
                            continue

                    train_X, test_X, ori_train_mask, ori_test_mask, train_num, test_num, train_cat_idx, test_cat_idx, train_mask, test_mask,cat_bin_num = load_dataset(dataname, mask_idx, mask_dir=mask_dir, DATA_DIR=DATA_DIR)

                    print('mask_idx:', mask_idx, 'sample_size:', sample_size, 'missing_rate:', missing_rate, 'mask_type:', mask_type, 'dataname:', dataname)
                    
                    ori_num = train_X.shape[0]

                    if benchmark_sample_size:
                        row_idx = subset_idx[sample_size]
                    else:
                        row_idx = np.arange(train_X.shape[0])

                    train_X, ori_train_mask, train_num, train_mask = train_X[row_idx], ori_train_mask[row_idx], train_num[row_idx], train_mask[row_idx]
                    
                    if train_cat_idx is not None:
                       train_cat_idx = train_cat_idx[row_idx] 

                    print('ori sample num:', ori_num, 'subset sample num:', train_X.shape[0])

                    # normalize data
                    mean_train_X, mean_test_X = train_X.mean(0), test_X.mean(0)
                    std_train_X, std_test_X = train_X.std(0), test_X.std(0)
                    # avoid division by zero
                    std_train_X, std_test_X = np.where(std_train_X == 0, 1e-3, std_train_X), np.where(std_test_X == 0, 1e-3, std_test_X)
                    train_X = (train_X - mean_train_X) / std_train_X 
                    #test_X = (test_X - mean_test_X) / std_test_X 
                    test_X = (test_X - mean_train_X) / std_train_X

                    # apply mask
                    x_train_miss = np.copy(train_X)
                    x_test_miss = np.copy(test_X)
                    x_train_miss[train_mask] = np.nan
                    x_test_miss[test_mask] = np.nan

                    # convert to torch tensor
                    X_train_true_np = np.copy(train_X)
                    X_test_true_np = np.copy(test_X)
                    X_train_miss = torch.from_numpy(x_train_miss).double()
                    X_test_miss = torch.from_numpy(x_test_miss).double()
                    X_train_true = torch.from_numpy(train_X).double()
                    X_test_true = torch.from_numpy(test_X).double()
                    
                    if not benchmark_sample_size and not benchmark_mask_ratio:
                        ckpt_train = f'OT_output/{mask_type}/filled_X/{dataname}/mask_{mask_idx}/train/'
                        os.makedirs(ckpt_train) if not os.path.exists(ckpt_train) else None

                        ckpt_test = f'OT_output/{mask_type}/filled_X/{dataname}/mask_{mask_idx}/test/'
                        os.makedirs(ckpt_test) if not os.path.exists(ckpt_test) else None

                        ckpt_in_sample = f'OT_output/{mask_type}/in_sample_result/{dataname}/'
                        os.makedirs(ckpt_in_sample) if not os.path.exists(ckpt_in_sample) else None

                        ckpt_out_sample = f'OT_output/{mask_type}/out_sample_result/{dataname}/'
                        os.makedirs(ckpt_out_sample) if not os.path.exists(ckpt_out_sample) else None
                    
                    elif benchmark_sample_size:
                        ckpt_train = f'OT_output/sensitive_analysis/sample_ratio/sample_size{sample_size}/{mask_type}/filled_X/{dataname}/mask_{mask_idx}/train/'
                        os.makedirs(ckpt_train) if not os.path.exists(ckpt_train) else None

                        ckpt_test = f'OT_output/sensitive_analysis/sample_ratio/sample_size{sample_size}/{mask_type}/filled_X/{dataname}/mask_{mask_idx}/test/'
                        os.makedirs(ckpt_test) if not os.path.exists(ckpt_test) else None

                        ckpt_in_sample = f'OT_output/sensitive_analysis/sample_ratio/sample_size{sample_size}/{mask_type}/in_sample_result/{dataname}/'
                        os.makedirs(ckpt_in_sample) if not os.path.exists(ckpt_in_sample) else None

                        ckpt_out_sample = f'OT_output/sensitive_analysis/sample_ratio/sample_size{sample_size}/{mask_type}/out_sample_result/{dataname}/'
                        os.makedirs(ckpt_out_sample) if not os.path.exists(ckpt_out_sample) else None
                    
                    elif benchmark_mask_ratio:
                        ckpt_train = f'OT_output/sensitive_analysis/missing_ratio/rate{missing_rate}/{mask_type}/filled_X/{dataname}/mask_{mask_idx}/train/'
                        os.makedirs(ckpt_train) if not os.path.exists(ckpt_train) else None

                        ckpt_test = f'OT_output/sensitive_analysis/missing_ratio/rate{missing_rate}/{mask_type}/filled_X/{dataname}/mask_{mask_idx}/test/'
                        os.makedirs(ckpt_test) if not os.path.exists(ckpt_test) else None

                        ckpt_in_sample = f'OT_output/sensitive_analysis/missing_ratio/rate{missing_rate}/{mask_type}/in_sample_result/{dataname}/'
                        os.makedirs(ckpt_in_sample) if not os.path.exists(ckpt_in_sample) else None

                        ckpt_out_sample = f'OT_output/sensitive_analysis/missing_ratio/rate{missing_rate}/{mask_type}/out_sample_result/{dataname}/'
                        os.makedirs(ckpt_out_sample) if not os.path.exists(ckpt_out_sample) else None


                    '''
                    TDM: Transformed Distribution Matching for Missing Value Imputation
                    '''
                    if not os.path.exists(f'{ckpt_train}/TDM.npy') or retrain:
                        args = {'out_dir': None, 
                                'niter': N_ITER,
                                'batchsize': 128, 
                                'lr': 1e-2, 
                                'network_width': 2, 
                                'network_depth': 3, 
                                'report_interval': 1000}

                        print('Begin TDM training...')
                        result = run_TDM(X_train_miss, args, X_train_true, imputer='TDM', return_imputer=False)
                        print('TDM training finished.')
                        TDM_X_filled = result['filled_X']

                        TDM_pred_X = TDM_X_filled[:]
                        len_num = train_num.shape[1]
                        res = TDM_pred_X[:, len_num:] * std_train_X[len_num:] + mean_train_X[len_num:]
                        TDM_pred_X[:, len_num:] = res
                        
                        TDM_mae, TDM_rmse, TDM_acc = 0.0,0.0,0.0
                        if is_cat_dataset:
                            TDM_mae, TDM_rmse, TDM_acc = get_eval(dataname, TDM_pred_X, X_train_true_np, train_cat_idx, train_num.shape[1], cat_bin_num, ori_train_mask)
                        else:
                            TDM_mae = result['learning_MAEs'][-1]
                            TDM_rmse = result['learning_RMSEs'][-1]

                        print('Saving TDM results...') 
                        # 1. save imputation   
                        np.save(f'{ckpt_train}/TDM.npy', TDM_X_filled) 
                        # 2. write in_sample performance
                        with open(f'{ckpt_in_sample}/in_sample_result_mask{mask_idx}.txt', 'a') as file: 
                            file.write(f"TDM in sample MAE: {TDM_mae}\n")
                            file.write(f"TDM in sample RMSE: {TDM_rmse}\n")
                            file.write(f"TDM in sample acc: {TDM_acc}\n")
                    
                    

                    # '''
                    # lin_RR_TDM: linear round-robin TDM
                    # '''
                    # if not os.path.exists(f'{ckpt_train}/lin_RR_TDM.npy') or retrain:
                    #     args = {'out_dir': None, 
                    #             'niter': NITER_RR, #15,
                    #             'batchsize': 128,
                    #             'rr_lr': 1e-2,
                    #             'proj_lr': 1e-2, 
                    #             'network_width': 2, 
                    #             'network_depth': 3, 
                    #             'max_niter': MAX_ITER, #10, 
                    #             'n_pairs': 10,
                    #             'report_interval': 1}

                    #     print('Begin lin_RR_TDM training...')
                    #     in_sample_result, lin_RR_TDM_imputer = run_TDM(X_train_miss, args, X_train_true, imputer='RR_TDM', return_imputer=True, rr_model='linear')
                    #     print('lin_RR_TDM training finished.')
                    #     lin_RR_TDM_X_filled = in_sample_result['filled_X']
                    #     lin_RR_TDM_mae, lin_RR_TDM_rmse, lin_RR_TDM_acc = 0.0,0.0,0.0
                        
                    #     if is_cat_dataset:
                    #         lin_RR_TDM_mae, lin_RR_TDM_rmse, lin_RR_TDM_acc = get_eval(dataname, lin_RR_TDM_X_filled, X_train_true_np, train_cat_idx, train_num.shape[1], cat_bin_num, ori_train_mask)
                    #     else:
                    #         lin_RR_TDM_mae = in_sample_result['learning_MAEs'][-1]
                    #         lin_RR_TDM_rmse = in_sample_result['learning_RMSEs'][-1]

                    #     print('Saving lin_RR_TDM results...')
                    #     # 1. save train imputation   
                    #     np.save(f'{ckpt_train}/lin_RR_TDM.npy', lin_RR_TDM_X_filled)   
                    #     # 2. write in_sample performance 
                    #     with open(f'{ckpt_in_sample}/in_sample_result_mask{mask_idx}.txt', 'a') as file: 
                    #         file.write(f"TDM lin RR in sample MAE: {lin_RR_TDM_mae}\n")
                    #         file.write(f"TDM lin RR in sample RMSE: {lin_RR_TDM_rmse}\n")
                    #         file.write(f"TDM lin RR in sample acc: {lin_RR_TDM_acc}\n")
                    #     # 3. save test imputation
                    #     lin_TDM_mae, lin_TDM_rmse, lin_TDM_acc = 0.0,0.0,0.0
                    #     lin_TDM_imp, lin_TDM_mae, lin_TDM_rmse = lin_RR_TDM_imputer.transform(X_test_miss, mask=torch.from_numpy(test_mask), verbose=True, X_true=X_test_true) 
                    
                    #     np.save(f'{ckpt_test}/lin_RR_TDM.npy', lin_TDM_imp)
                    #     if is_cat_dataset:
                    #         #override mae,rmse 
                    #         lin_TDM_mae, lin_TDM_rmse, lin_TDM_acc = get_eval(dataname, lin_TDM_imp, X_test_true_np, test_cat_idx, test_num.shape[1], cat_bin_num, ori_test_mask)
                        
                    #     # 4. write out of sample performance
                    #     with open(f'{ckpt_out_sample}/out_of_sample_result_mask{mask_idx}.txt', 'a') as file: 
                    #         file.write(f"TDM lin RR out of sample MAE: {lin_TDM_mae}\n")
                    #         file.write(f'TDM lin RR out of sample RMSE:{lin_TDM_rmse}\n')
                    #         file.write(f"TDM lin RR out of sample acc: {lin_TDM_acc}\n")


                    # '''
                    # mlp_RR_TDM: mlp round-robin TDM
                    # '''
                    # if not os.path.exists(f'{ckpt_train}/mlp_RR_TDM.npy') or retrain:
                    #     args = {'out_dir': None, 
                    #             'niter': NITER_RR,
                    #             'batchsize': 128,
                    #             'rr_lr': 1e-2,
                    #             'proj_lr': 1e-2, 
                    #             'network_width': 2, 
                    #             'network_depth': 3, 
                    #             'max_niter': MAX_ITER, 
                    #             'n_pairs': 10,
                    #             'report_interval': 1}

                    #     print('Begin mlp_RR_TDM training...')
                    #     in_sample_result, mlp_RR_TDM_imputer = run_TDM(X_train_miss, args, X_train_true, imputer='RR_TDM', return_imputer=True, rr_model='mlp')
                    #     print('mlp_RR_TDM training finished.')

                    #     mlp_RR_TDM_X_filled = in_sample_result['filled_X']
                    #     mlp_RR_TDM_mae, mlp_RR_TDM_rmse, mlp_RR_TDM_acc = 0.0,0.0,0.0

                    #     #in_sample result
                    #     if is_cat_dataset:
                    #         mlp_RR_TDM_mae, mlp_RR_TDM_rmse, mlp_RR_TDM_acc = get_eval(dataname, mlp_RR_TDM_X_filled, X_train_true_np, train_cat_idx, train_num.shape[1], cat_bin_num, ori_train_mask)
                    #     else:
                    #         mlp_RR_TDM_mae = in_sample_result['learning_MAEs'][-1]
                    #         mlp_RR_TDM_rmse = in_sample_result['learning_RMSEs'][-1]

                    #     print('Saving mlp_RR_TDM retuls...')
                    #     # 1. save train imputation  
                    #     np.save(f'{ckpt_train}/mlp_RR_TDM.npy', mlp_RR_TDM_X_filled)  
                    #     # 2. write in_sample performance 
                    #     with open(f'{ckpt_in_sample}/in_sample_result_mask{mask_idx}.txt', 'a') as file: 
                    #         file.write(f"TDM mlp RR in sample MAE: {mlp_RR_TDM_mae}\n")
                    #         file.write(f"TDM mlp RR in sample RMSE: {mlp_RR_TDM_rmse}\n")
                    #         file.write(f"TDM mlp RR in sample acc: {mlp_RR_TDM_acc}\n")
                    #     # 3. save test imputation
                    #     mlp_TDM_mae, mlp_TDM_rmse, mlp_TDM_acc = 0.0,0.0,0.0
                    #     mlp_TDM_imp, mlp_TDM_mae, mlp_TDM_rmse = mlp_RR_TDM_imputer.transform(X_test_miss, mask=torch.from_numpy(test_mask), verbose=True, X_true=X_test_true) 
                    #     np.save(f'{ckpt_test}/mlp_RR_TDM.npy', mlp_TDM_imp)

                    #     if is_cat_dataset:
                    #         #out-of-sample result, override mae 
                    #         mlp_TDM_mae, mlp_TDM_rmse, mlp_TDM_acc = get_eval(dataname, mlp_TDM_imp, X_test_true_np, test_cat_idx, test_num.shape[1], cat_bin_num, ori_test_mask)
                            
                    #     # 4. write out of sample performance
                    #     with open(f'{ckpt_out_sample}/out_of_sample_result_mask{mask_idx}.txt', 'a') as file: 
                    #         file.write(f"TDM mlp RR out of sample MAE: {mlp_TDM_mae}\n")
                    #         file.write(f'TDM mlp RR out of sample RMSE:{mlp_TDM_rmse}\n')
                    #         file.write(f"TDM mlp RR out of sample acc: {mlp_TDM_acc}\n")

                    '''
                    Hyperparameters
                    '''
                    n, d = X_train_miss.shape
                    batchsize = 128 # If the batch size is larger than half the dataset's size,
                                    # it will be redefined in the imputation methods.
                    lr = 1e-2
                    epsilon = pick_epsilon(X_train_miss) # Set the regularization parameter as a multiple of the median distance, as per the paper.

                    '''
                    Alg1 of MOT: Sinkhorn Imputation
                    '''
                    if not os.path.exists(f'{ckpt_train}/sk.npy') or retrain:
                        print('Begin Sinkhorn imputation...')
                        sk_imputer = OTimputer(eps=epsilon, batchsize=batchsize, lr=lr, niter=N_ITER)
                        sk_imp, sk_maes, sk_rmses = sk_imputer.fit_transform(X_train_miss, verbose=True, report_interval=500, X_true=X_train_true)
                        print('Sinkhorn imputation finished.')
                        print('Saving Sinkhorn results...')
                        sk_mae, sk_rmse, sk_acc = 0.0,0.0,0.0

                        sk_pred_X = sk_imp[:]
                        len_num = train_num.shape[1]
                        res = sk_pred_X[:, len_num:] * std_train_X[len_num:] + mean_train_X[len_num:]
                        sk_pred_X[:, len_num:] = res

                        if is_cat_dataset:
                            sk_mae, sk_rmse, sk_acc = get_eval(dataname, sk_pred_X, X_train_true_np, train_cat_idx, train_num.shape[1], cat_bin_num, ori_train_mask)
                        else:
                            sk_mae = sk_maes[-1]
                            sk_rmse = sk_rmses[-1]
                        # 1. save train imputation
                        np.save(f'{ckpt_train}/sk.npy', sk_imp)
                        # 2. write in_sample performance
                        with open(f'{ckpt_in_sample}/in_sample_result_mask{mask_idx}.txt', 'a') as file:
                            file.write(f"MOT in sample MAE: {sk_mae}\n")
                            file.write(f"MOT in sample RMSE: {sk_rmse}\n")
                            file.write(f"MOT in sample acc: {sk_acc}\n")

                    '''
                    Alg3 of MOT: Linear round-robin
                    '''
                    #Create the imputation models
                    d_ = d - 1
                    models = {}

                    for i in range(d):
                            models[i] = nn.Linear(d_, 1)

                    #Create the imputer
                    if not os.path.exists(f'{ckpt_train}/lin_RR_MOT.npy') or retrain:
                        print('Begin Linear RR imputation...')
                        #Best params from HPO: 'max_iter': 10, 'n_iter': 14, 'lr': 0.014392779132901597
                        lin_rr_imputer = RRimputer(models, eps=epsilon, lr=lr, max_iter=MAX_ITER, niter=NITER_RR)
                        lin_RR_MOT_imp, lin_RR_MOT_maes, lin_RR_MOT_rmses = lin_rr_imputer.fit_transform(X_train_miss, verbose=True, X_true=X_train_true)
                        print('Linear RR imputation finished.')
                        lin_RR_MOT_mae, lin_RR_MOT_rmse, lin_RR_MOT_acc = 0.0,0.0,0.0

                        lin_RR_MOT_pred_X = lin_RR_MOT_imp[:]
                        len_num = train_num.shape[1]
                        res = lin_RR_MOT_pred_X[:, len_num:] * std_train_X[len_num:] + mean_train_X[len_num:]
                        lin_RR_MOT_pred_X[:, len_num:] = res
                        
                        if is_cat_dataset:
                            lin_RR_MOT_mae, lin_RR_MOT_rmse, lin_RR_MOT_acc = get_eval(dataname, lin_RR_MOT_pred_X, X_train_true_np, train_cat_idx, train_num.shape[1], cat_bin_num, ori_train_mask)
                        else:
                            lin_RR_MOT_mae = lin_RR_MOT_maes[-1]
                            lin_RR_MOT_rmse = lin_RR_MOT_rmses[-1]

                        print('Saving MOT Linear RR results...')
                        # 1. save train imputation 
                        np.save(f'{ckpt_train}/lin_RR_MOT.npy', lin_RR_MOT_imp)
                        # 2. write in_sample performance
                        with open(f'{ckpt_in_sample}/in_sample_result_mask{mask_idx}.txt', 'a') as file:
                            file.write(f"MOT lin RR in sample MAE: {lin_RR_MOT_mae}\n")
                            file.write(f"MOT lin RR in sample RMSE: {lin_RR_MOT_rmse}\n")
                            file.write(f"MOT lin RR in sample acc: {lin_RR_MOT_acc}\n")
                        # 3. save test imputation
                        lin_mae, lin_rmse, lin_acc = 0.0, 0.0, 0.0
                        lin_imp, lin_mae, lin_rmse = lin_rr_imputer.transform(X_test_miss, mask=torch.from_numpy(test_mask), verbose=True, X_true=X_test_true)                      
                        np.save(f'{ckpt_test}/lin_RR_MOT.npy', lin_imp)

                        lin_mot_RR_pred_X = lin_imp[:]
                        len_num = test_num.shape[1]
                        res = lin_mot_RR_pred_X[:, len_num:] * std_train_X[len_num:] + mean_train_X[len_num:]
                        lin_mot_RR_pred_X[:, len_num:] = res
                        
                        if is_cat_dataset:
                            lin_mae, lin_rmse, lin_acc = get_eval(dataname, lin_mot_RR_pred_X, X_test_true_np, test_cat_idx, test_num.shape[1], cat_bin_num, ori_test_mask)

                        # 4. write out of sample performance
                        with open(f'{ckpt_out_sample}/out_of_sample_result_mask{mask_idx}.txt', 'a') as file:
                            file.write(f"MOT lin RR out of sample MAE: {lin_mae}\n")
                            file.write(f"MOT lin RR out of sample RMSE: {lin_rmse}\n")
                            file.write(f"MOT lin RR out of sample acc: {lin_acc}\n")

                    '''
                    Alg 3 of MOT: MLP round-robin 
                    '''
                    #Create the imputation models
                    d_ = d - 1
                    models = {}

                    for i in range(d):
                        models[i] = nn.Sequential(nn.Linear(d_, 2 * d_),
                                                nn.ReLU(),
                                                nn.Linear(2 * d_, d_),
                                                nn.ReLU(),
                                                nn.Linear(d_, 1))

                    #Create the imputer
                    if not os.path.exists(f'{ckpt_train}/mlp_RR_MOT.npy') or retrain:
                        print('Begin MLP RR imputation...')
                        mlp_rr_imputer = RRimputer(models, eps=epsilon, lr=lr, max_iter=MAX_ITER, niter=NITER_RR)
                        mlp_RR_MOT_imp, mlp_RR_MOT_maes, mlp_RR_MOT_rmses = mlp_rr_imputer.fit_transform(X_train_miss, verbose=True, X_true=X_train_true)
                        print('MLP RR imputation finished.')
                        mlp_RR_MOT_mae, mlp_RR_MOT_rmse, mlp_RR_MOT_acc = 0.0,0.0,0.0

                        mlp_RR_MOT_pred_X = mlp_RR_MOT_imp[:]
                        len_num = train_num.shape[1]
                        res = mlp_RR_MOT_pred_X[:, len_num:] * std_train_X[len_num:] + mean_train_X[len_num:]
                        mlp_RR_MOT_pred_X[:, len_num:] = res

                        if is_cat_dataset:
                            mlp_RR_MOT_mae, mlp_RR_MOT_rmse, mlp_RR_MOT_acc = get_eval(dataname, mlp_RR_MOT_pred_X, X_train_true_np, train_cat_idx, train_num.shape[1], cat_bin_num, ori_train_mask)
                        else:
                            mlp_RR_MOT_mae = mlp_RR_MOT_maes[-1]
                            mlp_RR_MOT_rmse = mlp_RR_MOT_rmses[-1]

                        print('Saving MOT MLP RR results...')
                        # 1. save train imputation
                        np.save(f'{ckpt_train}/mlp_RR_MOT.npy', mlp_RR_MOT_imp)
                        # 2. write in_sample performance
                        with open(f'{ckpt_in_sample}/in_sample_result_mask{mask_idx}.txt', 'a') as file:
                            file.write(f"MOT mlp RR in sample MAE: {mlp_RR_MOT_mae}\n")
                            file.write(f"MOT mlp RR in sample RMSE: {mlp_RR_MOT_rmse}\n")
                            file.write(f"MOT mlp RR in sample acc: {mlp_RR_MOT_acc}\n")
                        # 3. save test imputation
                        mlp_mae, mlp_rmse, mlp_acc = 0.0, 0.0, 0.0
                        mlp_imp, mlp_mae, mlp_rmse = mlp_rr_imputer.transform(X_test_miss, mask=torch.from_numpy(test_mask), verbose=True, X_true=X_test_true)
                        np.save(f'{ckpt_test}/mlp_RR_MOT.npy', mlp_imp)

                        mlp_mot_RR_pred_X = mlp_imp[:]
                        len_num = test_num.shape[1]
                        res = mlp_mot_RR_pred_X[:, len_num:] * std_train_X[len_num:] + mean_train_X[len_num:]
                        mlp_mot_RR_pred_X[:, len_num:] = res
                        
                        if is_cat_dataset: #override mae
                            mlp_mae, mlp_rmse, mlp_acc = get_eval(dataname, mlp_mot_RR_pred_X, X_test_true_np, test_cat_idx, test_num.shape[1], cat_bin_num, ori_test_mask)
                        
                        # 4. write out of sample performance
                        with open(f'{ckpt_out_sample}/out_of_sample_result_mask{mask_idx}.txt', 'a') as file:
                            file.write(f"MOT mlp RR out of sample MAE: {mlp_mae}\n")
                            file.write(f"MOT mlp RR out of sample RMSE: {mlp_rmse}\n")
                            file.write(f"MOT mlp RR out of sample acc: {mlp_acc}\n")

