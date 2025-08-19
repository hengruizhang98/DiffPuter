import os 
import sys
import os.path as osp
import numpy as np
import torch
import argparse
import pandas as pd
from sklearn import preprocessing

#from IGRM import train_gnn_mdi, add_uci_subparser, get_data_fix_mask, out_of_sample_test_gnn_mdi
from training.gnn_mdi import train_gnn_mdi, out_of_sample_test_gnn_mdi
from uci.uci_data import get_data_fix_mask
from uci.uci_subparser import add_uci_subparser
sys.path.append("..")
from data_utils import load_dataset, get_eval

import pdb

parser = argparse.ArgumentParser()

#parser.add_argument('--dataname', type=str, default='california', help='Name of dataset.')
parser.add_argument('--mask_num', type=int, default=10) 
parser.add_argument('--mask_type', type=str, default='MAR')
parser.add_argument('--missing_rate', type=float, default=0.3)
parser.add_argument('-retrain', action='store_true', help='Retrain the models or not')
parser.add_argument('--gpu', type=int, default=0, help='GPU index.')

parser.add_argument('--model_types', type=str, default='EGSAGE_EGSAGE_EGSAGE')
parser.add_argument('--post_hiddens', type=str, default=None,) # default to be 1 hidden of node_dim
parser.add_argument('--concat_states', action='store_true', default=False)
parser.add_argument('--norm_embs', type=str, default=None,) # default to be all true
parser.add_argument('--aggr', type=str, default='mean',)
parser.add_argument('--node_dim', type=int, default=64)
parser.add_argument('--edge_dim', type=int, default=64)
parser.add_argument('--edge_mode', type=int, default=1)  # 0: use it as weight; 1: as input to mlp
parser.add_argument('--gnn_activation', type=str, default='relu')
parser.add_argument('--impute_hiddens', type=str, default='64')
parser.add_argument('--impute_activation', type=str, default='relu')
parser.add_argument('--epochs', type=int, default=20000)
parser.add_argument('--opt', type=str, default='adam')
parser.add_argument('--opt_scheduler', type=str, default='none')
parser.add_argument('--opt_restart', type=int, default=0)
parser.add_argument('--opt_decay_step', type=int, default=1000)
parser.add_argument('--opt_decay_rate', type=float, default=0.9)
parser.add_argument('--dropout', type=float, default=0.)
parser.add_argument('--weight_decay', type=float, default=0.)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--known', type=float, default=0.7) # 1 - edge dropout rate
parser.add_argument('--auto_known', action='store_true', default=False)
parser.add_argument('--loss_mode', type=int, default = 0) # 0: loss on all train edge, 1: loss only on unknown train edge
parser.add_argument('--valid', type=float, default=0.) # valid-set ratio
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--log_dir', type=str, default='0')
parser.add_argument('--save_model', action='store_true', default=False)
parser.add_argument('--save_prediction', action='store_true', default=False)
parser.add_argument('--transfer_dir', type=str, default=None)
parser.add_argument('--transfer_extra', type=str, default='')
parser.add_argument('--mode', type=str, default='train') # debug

parser.add_argument('--train_edge', type=float, default=0.7)
parser.add_argument('--split_sample', type=float, default=0.)
parser.add_argument('--split_by', type=str, default='y') # 'y', 'random'
parser.add_argument('--split_train', action='store_true', default=False)
parser.add_argument('--split_test', action='store_true', default=False)
parser.add_argument('--train_y', type=float, default=0.7)

parser.add_argument('--method', type=str, default='mean')
parser.add_argument('--level', type=int, default=0)

parser.add_argument('--best_level', action='store_true', default=False)
parser.add_argument('--comment', type=str, default='v1')


subparsers = parser.add_subparsers()
add_uci_subparser(subparsers)
args = parser.parse_args()
print(args)

# check cuda
if args.gpu != -1 and torch.cuda.is_available():
    args.device = f'cuda:{args.gpu}'
else:
    args.device = 'cpu'
print('Using device:', args.device)
device = args.device

if __name__ == '__main__':

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # exclude adult, beijing, default, news, which will OOM
    datanames = ['adult','california', 'gesture', 'letter', 'magic', 'shoppers','bean'] 
    
    retrain = args.retrain
    mask_num = args.mask_num
    mask_type = args.mask_type
    missing_rate = args.missing_rate
    
    for dataname in datanames:

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
            print('mask_idx:', mask_idx)

            ckpt_train = f'IGRM_output/{mask_type}/filled_X/{dataname}/mask_{mask_idx}/train/'
            os.makedirs(ckpt_train) if not os.path.exists(ckpt_train) else None

            ckpt_test = f'IGRM_output/{mask_type}/filled_X/{dataname}/mask_{mask_idx}/test/'
            os.makedirs(ckpt_test) if not os.path.exists(ckpt_test) else None

            ckpt_in_sample = f'IGRM_output/{mask_type}/in_sample_result/{dataname}/'
            os.makedirs(ckpt_in_sample) if not os.path.exists(ckpt_in_sample) else None

            ckpt_out_sample = f'IGRM_output/{mask_type}/out_sample_result/{dataname}/'
            os.makedirs(ckpt_out_sample) if not os.path.exists(ckpt_out_sample) else None

            if os.path.exists(f'{ckpt_test}/IGRM.npy'):
                continue

            train_X, test_X, ori_train_mask, ori_test_mask, train_num, test_num, train_cat_idx, test_cat_idx, train_mask, test_mask,cat_bin_num = load_dataset(dataname, mask_idx, mask_dir=mask_dir)
            
            # 0 mean unit variance normalization
            mean_train_X, mean_test_X = train_X.mean(0), test_X.mean(0)
            std_train_X, std_test_X = train_X.std(0), test_X.std(0)
            # avoid division by zero
            std_train_X, std_test_X = np.where(std_train_X == 0, 1e-3, std_train_X), np.where(std_test_X == 0, 1e-3, std_test_X)
            train_X_norm = (train_X - mean_train_X) / std_train_X 
            #test_X = (test_X - mean_test_X) / std_test_X 
            test_X_norm = (test_X - mean_train_X) / std_train_X
            X_train_true_norm_np = np.copy(train_X_norm)
            X_test_true_norm_np = np.copy(test_X_norm)

            # apply minmaxscaler normalize for train and test
            min_max_scaler_train = preprocessing.MinMaxScaler()
            train_X = min_max_scaler_train.fit_transform(train_X)

            min_max_scaler_test = preprocessing.MinMaxScaler()
            test_X = min_max_scaler_test.fit_transform(test_X)
            
            if not hasattr(args,'split_sample'):
                args.split_sample = 0
            df_X = pd.DataFrame(train_X) # for compatibility with IGRM
            df_X_test = pd.DataFrame(test_X)
            data = get_data_fix_mask(df_X, train_mask, confidence=0.6)
            
            data_test = get_data_fix_mask(df_X_test, test_mask, confidence=0.6)

            log_path = f'IGRM_output/filled_X/{dataname}/mask_{mask_idx}/cmd_input/'
            os.makedirs(log_path) if not osp.exists(log_path) else None

            cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
            with open(osp.join(log_path, 'cmd_input.txt'), 'a') as f:
                f.write(cmd_input)

            X_filled, impute_model, model = train_gnn_mdi(data, args, log_path, device, return_filled_X=True)
            
            # reverse minmaxscaler normalization
            X_filled = min_max_scaler_train.inverse_transform(X_filled)

            # apply zero mean unit var normalize 
            mean_X_filled = X_filled.mean(0)
            std_X_filled = X_filled.std(0)
            # avoid division by zero
            std_X_filled = np.where(std_X_filled == 0, 1e-3, std_X_filled)
            #X_filled = (X_filled - mean_X_filled) / std_X_filled 
            
            len_num = train_num.shape[1]
            X_filled_train_num = X_filled[:, :len_num].copy()
            X_filled_train_num = (X_filled_train_num - mean_X_filled[:len_num]) / std_X_filled[:len_num]
            X_filled[:, :len_num] = X_filled_train_num
                
            mae, rmse, acc = get_eval(dataname, X_filled, X_train_true_norm_np, train_cat_idx, train_num.shape[1], cat_bin_num, ori_train_mask)
            
            print('Saving IGRM in sample results...')  
            # 1. save imputation   
            np.save(f'{ckpt_train}/IGRM.npy', X_filled) 
            # 2. write in_sample performance
            with open(f'{ckpt_in_sample}/in_sample_result_mask{mask_idx}.txt', 'w') as file: 
                file.write(f"IGRM in sample MAE: {mae}\n")
                file.write(f"IGRM in sample RMSE: {rmse}\n")
                file.write(f"IGRM in sample acc: {acc}\n")
            
            # ================================================================================================
            # Out of sample evaluation

            X_filled_test = out_of_sample_test_gnn_mdi(data_test, impute_model, model, log_path, device, return_filled_X=True)
            
            # reverse minmaxscaler normalization
            X_filled_test = min_max_scaler_test.inverse_transform(X_filled_test)

            # apply zero mean unit var normalize 
            mean_X_filled_test = X_filled_test.mean(0)
            std_X_filled_test = X_filled_test.std(0)

            # avoid division by zero
            std_X_filled_test = np.where(std_X_filled_test == 0, 1e-3, std_X_filled_test)
             
            len_num = test_num.shape[1]
            X_filled_test_num = X_filled_test[:, :len_num].copy()
            X_filled_test_num = (X_filled_test_num - mean_X_filled_test[:len_num]) / std_X_filled_test[:len_num]
            X_filled_test[:, :len_num] = X_filled_test_num
            
            mae_test, rmse_test, acc_test = get_eval(dataname, X_filled_test, X_test_true_norm_np, test_cat_idx, test_num.shape[1], cat_bin_num, ori_test_mask)
            
            print('Saving IGRM out of sample results...')  
            # 1. save imputation   
            np.save(f'{ckpt_test}/IGRM.npy', X_filled_test) 
            # 2. write out of sample performance
            with open(f'{ckpt_out_sample}/out_sample_result_mask{mask_idx}.txt', 'w') as file: 
                file.write(f"IGRM out of sample MAE: {mae_test}\n")
                file.write(f"IGRM out of sample RMSE: {rmse_test}\n")
                file.write(f"IGRM out of sample acc: {acc_test}\n")

