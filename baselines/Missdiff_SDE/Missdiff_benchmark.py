import os
import torch

import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import warnings
import time

import sys
from tqdm import tqdm
from model import MLPDiffusion, Model
from diffusion_utils import sample_step, impute_mask
sys.path.append("..")
from data_utils import load_dataset, get_eval
import pdb

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Missing Value Imputation')

parser.add_argument('--dataname', type=str, default='california', help='Name of dataset.')
parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
parser.add_argument('--split_idx', type=int, default=0, help='Split idx.')
parser.add_argument('--hid_dim', type=int, default=1024, help='Hidden dimension.')
parser.add_argument('--num_epochs', type=int, default=10000+1, help='Number of epochs.')
#parser.add_argument('--mask_type', type=str, default='MCAR', help='Type of mask.')
parser.add_argument('--missing_rate', type=float, default=0.3, help='Missing rate.')
parser.add_argument('--init', type=str, default='zero', help='Initialization method.')
parser.add_argument('--mask_num', type=int, default=10, help='Number of masks.')

args = parser.parse_args()

# check cuda
if args.gpu != -1 and torch.cuda.is_available():
    args.device = f'cuda:{args.gpu}'
else:
    args.device = 'cpu'

datanames = ['bean', 'adult', 'beijing', 'california', 'default', 'gesture', 'letter', 'magic', 'news', 'shoppers'] 


if __name__ == '__main__':
    for mask_type in ['MCAR', 'MAR', 'MNAR']:
        dataname = args.dataname
        split_idx = args.split_idx
        device = args.device
        hid_dim = args.hid_dim
        num_epochs = args.num_epochs
        init_method = args.init

        mask_num = args.mask_num
        missing_rate = args.missing_rate

        DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'datasets'))

        for dataname in datanames:
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
        
            for split_idx in range(mask_num):

                train_X, test_X, ori_train_mask, ori_test_mask, train_num, test_num, train_cat_idx, test_cat_idx, train_mask, test_mask,cat_bin_num = load_dataset(dataname, split_idx, mask_dir=mask_dir)

                ckpt_train = f'missdiff_sde_output/{mask_type}/filled_X/{dataname}/mask_{split_idx}/train/{init_method}/'
                os.makedirs(ckpt_train) if not os.path.exists(ckpt_train) else None

                ckpt_test = f'missdiff_sde_output/{mask_type}/filled_X/{dataname}/mask_{split_idx}/test/{init_method}/'
                os.makedirs(ckpt_test) if not os.path.exists(ckpt_test) else None

                ckpt_eval_sample = f'missdiff_sde_output/{mask_type}/eval_result/{dataname}/{init_method}/'
                os.makedirs(ckpt_eval_sample) if not os.path.exists(ckpt_eval_sample) else None

                if os.path.exists(f'{ckpt_eval_sample}/result_mask{split_idx}.txt'):
                    print(f'{ckpt_eval_sample}/result_mask{split_idx}.txt exists, skip')
                    continue
                
                mean_X = train_X.mean(0)
                std_X = train_X.std(0)
                in_dim = train_X.shape[1]

                X = (train_X - mean_X) / std_X / 2
                X = torch.tensor(X)

                X_test = (test_X - mean_X) / std_X / 2
                X_test = torch.tensor(X_test)

                mask_train = torch.tensor(train_mask)
                mask_test = torch.tensor(test_mask)

                MAEs = []
                RMSEs = []
                ACCs = []

                MAEs_out = []
                RMSEs_out = []
                ACCs_out = []

                ckpt_dir = f'ckpt/{dataname}/{split_idx}/'
                os.makedirs(ckpt_dir) if not os.path.exists(ckpt_dir) else None
                
                if init_method == 'zero':
                    X_miss = (1. - mask_train.float()) * X
                    train_data = X_miss.numpy()
                elif init_method == 'mean': 
                    # use the mean of each column to fill the missing values
                    X_miss = X.numpy().copy()
                    X_miss[train_mask] = np.nan
                    mean_ret = np.nanmean(X_miss, axis = 0)
                    # use mean_ret to fill the missing values
                    for i in range(X_miss.shape[1]):
                        X_miss[np.isnan(X_miss[:,i]), i] = mean_ret[i]
                    train_data = X_miss

                elif init_method == 'remasker':
                    raise NotImplementedError
                    train_data = np.load(f'../Remasker_output_new/{mask_type}/filled_X/{dataname}/mask_{split_idx}/train/remasker.npy')    
                     
                # Combine the data and mask, keep track of missing entities.
                comb_train_data = np.concatenate([train_data, mask_train.numpy()], axis = 1)
                
                batch_size = 4096
                train_loader = DataLoader(
                    comb_train_data,
                    batch_size = batch_size,
                    shuffle = True,
                    num_workers = 4,
                )

                denoise_fn = MLPDiffusion(in_dim, hid_dim).to(device)
                print(denoise_fn)

                num_params = sum(p.numel() for p in denoise_fn.parameters())
                print("the number of parameters", num_params)

                model = Model(denoise_fn = denoise_fn, hid_dim = in_dim).to(device)

                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)
                scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=20, verbose=True)

                model.train()

                best_loss = float('inf')
                patience = 0
                start_time = time.time()
                for epoch in range(num_epochs):

                    pbar = tqdm(train_loader, total=len(train_loader))
                    pbar.set_description(f"Epoch {epoch+1}/{num_epochs}")

                    batch_loss = 0.0
                    len_input = 0
                    for batch in pbar:
                        inputs = batch.float().to(device)
                        loss = model(inputs)

                        loss = loss.mean()
                        batch_loss += loss.item() * len(inputs)
                        len_input += len(inputs)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        pbar.set_postfix({"Loss": loss.item()})

                    curr_loss = batch_loss/len_input
                    scheduler.step(curr_loss)

                    if curr_loss < best_loss:
                        best_loss = loss.item()
                        patience = 0
                        torch.save(model.state_dict(), f'{ckpt_dir}/model.pt')
                    else:
                        patience += 1
                        if patience == 1000:
                            print('Early stopping')
                            break

                    if epoch % 1000 == 0:
                        torch.save(model.state_dict(), f'{ckpt_dir}/model_{epoch}.pt')

                    end_time = time.time()

                # In-sample imputation
                num_trials = 10

                rec_Xs = []

                for trial in range(num_trials):
                    
                    X_miss = (1. - mask_train.float()) * X
                    X_miss = X_miss.to(device)
                    impute_X = X_miss

                    in_dim = X.shape[1]

                    denoise_fn = MLPDiffusion(in_dim, hid_dim).to(device)

                    ckpt_dir = f'ckpt/{dataname}'
                    model = Model(denoise_fn = denoise_fn, hid_dim = in_dim).to(device)
                    model.load_state_dict(torch.load(f'{ckpt_dir}/{split_idx}/model.pt'))

                    # ==========================================================

                    num_steps = 50
                    net = model.denoise_fn_D

                    num_samples, dim = X.shape[0], X.shape[1]
                    rec_X = impute_mask(net, impute_X, mask_train, num_samples, dim, num_steps, device)
                    
                    mask_int = mask_train.to(torch.float).to(device)
                    rec_X = rec_X * mask_int + impute_X * (1-mask_int)
                    rec_Xs.append(rec_X)
                    
                    print(f'Trial = {trial}')

                rec_X = torch.stack(rec_Xs, dim = 0).mean(0) 

                rec_X = rec_X.cpu().numpy() * 2
                X_true = X.cpu().numpy() * 2

                np.save(f'{ckpt_train}/missdiff.npy', rec_X) 

                pred_X = rec_X[:]
                len_num = train_num.shape[1]
                res = pred_X[:, len_num:] * std_X[len_num:] + mean_X[len_num:]
                pred_X[:, len_num:] = res

                mae, rmse, acc = get_eval(dataname, pred_X, X_true, train_cat_idx, train_num.shape[1], cat_bin_num, ori_train_mask)
                MAEs.append(mae)
                RMSEs.append(rmse)
                ACCs.append(acc)

                print('in-sample',mae, rmse, acc)


                # out-of_sample_imputation

                num_trials = 10

                rec_Xs = []

                for trial in range(num_trials):
                    
                    # For out-of-sample imputation, no results from previous iterations are used

                    X_miss = (1. - mask_test.float()) * X_test
                    X_miss = X_miss.to(device)
                    impute_X = X_miss

                    in_dim = X_test.shape[1]

                    denoise_fn = MLPDiffusion(in_dim, hid_dim).to(device)

                    ckpt_dir = f'ckpt/{dataname}'
                    model = Model(denoise_fn = denoise_fn, hid_dim = in_dim).to(device)
                    model.load_state_dict(torch.load(f'{ckpt_dir}/{split_idx}/model.pt'))

                    # ==========================================================

                    num_steps = 50
                    net = model.denoise_fn_D

                    num_samples, dim = X_test.shape[0], X_test.shape[1]
                    rec_X = impute_mask(net, impute_X, mask_test, num_samples, dim, num_steps, device)
                    
                    mask_int = mask_test.to(torch.float).to(device)
                    rec_X = rec_X * mask_int + impute_X * (1-mask_int)
                    rec_Xs.append(rec_X)
                    
                    print(f'Trial = {trial}')

                rec_X = torch.stack(rec_Xs, dim = 0).mean(0) 

                rec_X = rec_X.cpu().numpy() * 2
                X_true = X_test.cpu().numpy() * 2

                pred_X = rec_X[:]
                len_num = train_num.shape[1]
                res = pred_X[:, len_num:] * std_X[len_num:] + mean_X[len_num:]
                pred_X[:, len_num:] = res

                mae_out, rmse_out, acc_out = get_eval(dataname, pred_X, X_true, test_cat_idx, test_num.shape[1], cat_bin_num, ori_test_mask)
                MAEs_out.append(mae_out)
                RMSEs_out.append(rmse_out)
                ACCs_out.append(acc_out)

                with open (f'{ckpt_eval_sample}/result_mask{split_idx}.txt', 'a+') as f:

                    f.write(f'MAE: in-sample: {mae}, out-of-sample: {mae_out} \n')
                    f.write(f'RMSE: in-sample: {rmse}, out-of-sample: {rmse_out} \n')
                    f.write(f'ACC: in-sample: {acc}, out-of-sample: {acc_out} \n')

                print('out-of-sample', mae, rmse, acc)
