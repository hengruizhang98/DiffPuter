"""
Official implementation of MCFlow -
"""
import numpy as np
import torch
import argparse
import sys
import os
from models import InterpRealNVP
import util
from loader import DataLoader
from models import LatentToLatentApprox
sys.path.append("..")
from data_utils import load_dataset, get_eval

import pdb

def main(args):
    datanames = ['bean','california', 'adult', 'beijing' , 'default', 'gesture', 'letter', 'magic', 'news', 'shoppers'] 
    retrain = args.retrain
    mask_num = args.mask_num
    mask_type = args.mask_type
    missing_rate = args.missing_rate

    for dataname in datanames:
        print('dataname:', dataname)
        
        DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'datasets'))

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
            train_X, test_X, ori_train_mask, ori_test_mask, train_num, test_num, train_cat_idx, test_cat_idx, train_mask, test_mask, cat_bin_num = load_dataset(dataname, split_idx, mask_dir=mask_dir)
            
            # 0 mean unit variance normalization
            mean_train_X = train_X.mean(0)
            std_train_X = train_X.std(0)
           
            std_train_X = np.where(std_train_X == 0, 1e-3, std_train_X) #avoid division by zero
            
            train_X_norm = (train_X - mean_train_X) / std_train_X 
            test_X_norm = (test_X - mean_train_X) / std_train_X

            X_train_true_norm_np = np.copy(train_X_norm)
            X_test_true_norm_np = np.copy(test_X_norm)

            ckpt_path = f'ckpt/{dataname}/{mask_type}/{split_idx}' #for .pt model saving
            os.makedirs(ckpt_path) if not os.path.exists(ckpt_path) else None
            
            ckpt_train = f'MCFlow_output/{mask_type}/filled_X/{dataname}/mask_{split_idx}/train/' #for .npy filled_X saving
            os.makedirs(ckpt_train) if not os.path.exists(ckpt_train) else None

            ckpt_test = f'MCFlow_output/{mask_type}/filled_X/{dataname}/mask_{split_idx}/test/' #for .npy filled_X saving
            os.makedirs(ckpt_test) if not os.path.exists(ckpt_test) else None

            ckpt_eval_sample = f'MCFlow_output/{mask_type}/eval_result/{dataname}/' #for eval result saving
            os.makedirs(ckpt_eval_sample) if not os.path.exists(ckpt_eval_sample) else None

            if os.path.exists(f'{ckpt_test}/MCFlow.npy') and not retrain:
                print(f'{ckpt_test} already exists, skipping...')
                continue

            #initialize dataset class
            ldr = DataLoader(mode=0, seed=args.seed, path=args.dataset, train_X=train_X, test_X=test_X, train_mask=train_mask, test_mask=test_mask, drp_percent=args.drp_impt)
            train_data_loader = torch.utils.data.DataLoader(ldr, batch_size=args.batch_size, shuffle=True, drop_last=False)
            eval_data_loader = torch.utils.data.DataLoader(ldr, batch_size=args.batch_size, shuffle=False, drop_last=False)
            num_neurons = int(ldr.train[0].shape[0]) #60 for news, 784 for mnist

            #Initialize normalizing flow model neural network and its optimizer
            flow = util.init_flow_model(num_neurons, args.num_nf_layers, InterpRealNVP, ldr.train[0].shape[0], args)
            nf_optimizer = torch.optim.Adam([p for p in flow.parameters() if p.requires_grad==True], lr=args.lr)

            #Initialize latent space neural network and its optimizer
            num_hidden_neurons = [int(ldr.train[0].shape[0]), int(ldr.train[0].shape[0]), int(ldr.train[0].shape[0]), int(ldr.train[0].shape[0]),  int(ldr.train[0].shape[0])]
            nn_model = LatentToLatentApprox(int(ldr.train[0].shape[0]), num_hidden_neurons).float()
            if args.use_cuda:
                nn_model.cuda()
            nn_optimizer = torch.optim.Adam([p for p in nn_model.parameters() if p.requires_grad==True], lr=args.lr)

            reset_scheduler = 2

            print("\n*********************************")
            print(f"Starting {dataname} experiment\n")

            #Train and test MCFlow
            for epoch in range(args.n_epochs):
                util.endtoend_train(flow, nn_model, nf_optimizer, nn_optimizer, train_data_loader, args, epoch) #Train the MCFlow model

                with torch.no_grad():
                    ldr.mode=1 #Use testing data
                    te_mse, _ = util.endtoend_test(flow, nn_model, train_data_loader, args) #Test MCFlow model
                    ldr.mode=0 #Use training data
                    print("Epoch", epoch, " Test RMSE", te_mse**.5)

                if (epoch+1) % reset_scheduler == 0:
                    #Reset unknown values in the dataset using predicted estimates
                    if args.dataset == 'mnist':
                        ldr.reset_img_imputed_values(nn_model, flow, args.seed, args)
                    else:
                        ldr.reset_imputed_values(nn_model, flow, args.seed, args)
                    flow = util.init_flow_model(num_neurons, args.num_nf_layers, InterpRealNVP, ldr.train[0].shape[0], args) #Initialize brand new flow model to train on new dataset
                    nf_optimizer = torch.optim.Adam([p for p in flow.parameters() if p.requires_grad==True], lr=args.lr)
                    reset_scheduler = reset_scheduler*2
            
            ldr.mode=0 #Use training data
            filled_X_train, _ = util.get_filled_data(flow, nn_model, eval_data_loader, args)
            filled_X_train = ldr.min_max_scaler_train.inverse_transform(filled_X_train)

            ldr.mode=1 #Use testing data
            filled_X_test, rmse_1 = util.get_filled_data(flow, nn_model, eval_data_loader, args)
            filled_X_test = ldr.min_max_scaler_test.inverse_transform(filled_X_test)
            
            # apply zero mean unit var normalize 
            mean_X_filled_train = filled_X_train.mean(0)
            std_X_filled_train = filled_X_train.std(0)
            std_X_filled_train = np.where(std_X_filled_train == 0, 1e-3, std_X_filled_train)
            len_num = train_num.shape[1]

            X_filled_train_num = filled_X_train[:, :len_num].copy()
            X_filled_train_num = (X_filled_train_num - mean_X_filled_train[:len_num]) / std_X_filled_train[:len_num]
            filled_X_train[:, :len_num] = X_filled_train_num
            

            mean_X_filled_test = filled_X_test.mean(0)
            std_X_filled_test = filled_X_test.std(0)
            std_X_filled_test = np.where(std_X_filled_test == 0, 1e-3, std_X_filled_test)
            
            X_filled_test_num = filled_X_test[:, :len_num].copy()
            X_filled_test_num = (X_filled_test_num - mean_X_filled_test[:len_num]) / std_X_filled_test[:len_num]
            filled_X_test[:, :len_num] = X_filled_test_num
            

            mae_train, rmse_train, acc_train = get_eval(dataname, filled_X_train, X_train_true_norm_np, train_cat_idx, train_num.shape[1], cat_bin_num, ori_train_mask)
            mae_test, rmse_test, acc_test = get_eval(dataname, filled_X_test, X_test_true_norm_np, test_cat_idx, test_num.shape[1], cat_bin_num, ori_test_mask)
                    
            print('Saving MCFlow out of sample results...')  
            # 1. save imputation   
            np.save(f'{ckpt_train}/MCFlow.npy', filled_X_train)
            np.save(f'{ckpt_test}/MCFlow.npy', filled_X_test) 

            # 2. save sample performance
            with open (f'{ckpt_eval_sample}/result_mask{split_idx}.txt', 'a+') as f:

                f.write(f'MAE: in-sample: {mae_train}, out-of-sample: {mae_test} \n')
                f.write(f'RMSE: in-sample: {rmse_train}, out-of-sample: {rmse_test} \n')
                f.write(f'ACC: in-sample: {acc_train}, out-of-sample: {acc_test} \n')

        
''' Run MCFlow experiment '''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='Reproducibility')
    parser.add_argument('--batch-size', type=int, default=64) #64
    parser.add_argument('--num-nf-layers', type=int, default=3)
    parser.add_argument('--n-epochs', type=int, default=300)
    parser.add_argument('--drp-impt', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--use-cuda', type=util.str2bool, default=True)
    parser.add_argument('--dataset', default='news', help='Two options: (1) letter-recogntion or (2) mnist')

    parser.add_argument('-retrain', action='store_true', help='Retrain the models or not')
    parser.add_argument('--mask_num', type=int, default=10)
    parser.add_argument('--mask_type', default='MCAR', help='Three options: (1) MCAR, (2) MAR, (3) MNAR')
    parser.add_argument('--missing_rate', type=float, default=0.3, help='Missing rate for MCAR, MAR, MNAR')

    args = parser.parse_args()

    ''' Reproducibility '''
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    ''' Cuda enabled experimentation check '''
    if not torch.cuda.is_available() or args.use_cuda==False:
        print("CUDA Unavailable. Using cpu. Check torch.cuda.is_available()")
        args.use_cuda = False

    if not os.path.exists('masks'):
        os.makedirs('masks')

    if not os.path.exists('data'):
        os.makedirs('data')

    main(args)
    print("Experiment completed")
