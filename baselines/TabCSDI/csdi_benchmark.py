import argparse
import torch
import datetime
import json
import yaml
import os
import numpy as np
import sys
import pdb

from src.main_model_table import TabCSDI
from src.utils_table import train, evaluate
from dataset_UCI import uic_tabular_dataset
from torch.utils.data import DataLoader
sys.path.append("..")
from data_utils import load_dataset, get_eval


parser = argparse.ArgumentParser(description="TabCSDI")
parser.add_argument("--config", type=str, default="uci.yaml")
parser.add_argument("--device", default="cpu", help="Device")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--testmissingratio", type=float, default=0.2)
parser.add_argument("--nfold", type=int, default=5, help="for 5-fold test")
parser.add_argument("--unconditional", action="store_true", default=0)
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=3) #100
parser.add_argument("--mask_num", type=int, default=10, help="number of masks")

args = parser.parse_args()
print(args)

datanames = ['bean', 'adult', 'beijing', 'california', 'default', 'gesture', 'letter', 'magic', 'news', 'shoppers'] 
mask_types = ['MCAR', 'MAR', 'MNAR']
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'datasets'))

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

if __name__ == "__main__":
    for dataname in datanames:
        for mask_type in mask_types:
            for mask_idx in range(args.mask_num):
                ckpt_train = f'TabCSDI_output/{mask_type}/filled_X/{dataname}/mask_{mask_idx}/train/'
                os.makedirs(ckpt_train) if not os.path.exists(ckpt_train) else None

                ckpt_test = f'TabCSDI_output/{mask_type}/filled_X/{dataname}/mask_{mask_idx}/test/'
                os.makedirs(ckpt_test) if not os.path.exists(ckpt_test) else None

                ckpt_in_sample = f'TabCSDI_output/{mask_type}/in_sample_result/{dataname}/'
                os.makedirs(ckpt_in_sample) if not os.path.exists(ckpt_in_sample) else None

                ckpt_out_sample = f'TabCSDI_output/{mask_type}/out_sample_result/{dataname}/'
                os.makedirs(ckpt_out_sample) if not os.path.exists(ckpt_out_sample) else None

                print(f'Processing {dataname} with {mask_type} mask {mask_idx}...')
                if os.path.exists(f'{ckpt_in_sample}/in_sample_result_mask{mask_idx}.txt'):
                    print(f'{ckpt_in_sample}/in_sample_result_mask{mask_idx}.txt already exists, skipping...')
                    continue 

                path = "config/" + args.config
                with open(path, "r") as f:
                    config = yaml.safe_load(f)

                config["model"]["is_unconditional"] = args.unconditional
                config["model"]["test_missing_ratio"] = args.testmissingratio

                print(json.dumps(config, indent=4))

                # Create folder
                current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                foldername = "./save/breast_fold" + str(args.nfold) + "_" + current_time + "/"
                print("model folder:", foldername)
                os.makedirs(foldername, exist_ok=True)
                with open(foldername + "config.json", "w") as f:
                    json.dump(config, f, indent=4)
                
                folder_name = 'rate30' #tmp
                if mask_type == 'MCAR':
                    mask_dir = f'{DATA_DIR}/{dataname}/masks/{folder_name}/MCAR' 
                elif mask_type == 'MAR':
                    mask_dir = f'{DATA_DIR}/{dataname}/masks/{folder_name}/MAR'
                elif mask_type == 'MNAR':
                    mask_dir = f'{DATA_DIR}/{dataname}/masks/{folder_name}/MNAR_logistic_T2'
                else:
                    raise ValueError('Invalid mask type, please choose from MCAR, MAR, MNAR.')

                train_X, test_X, ori_train_mask, ori_test_mask, train_num, test_num, train_cat_idx, test_cat_idx, train_mask, test_mask,cat_bin_num = load_dataset(dataname, mask_idx, mask_dir=mask_dir)
                
                # normalize gt for evaluation
                X_train_true_np = np.copy(train_X)
                X_test_true_np = np.copy(test_X)

                mean_train_X = X_train_true_np.mean(0)
                std_train_X = X_train_true_np.std(0)

                std_train_X = np.where(std_train_X == 0, 1e-3, std_train_X)

                X_train_true_np = (X_train_true_np - mean_train_X) / std_train_X 
                X_test_true_np = (X_test_true_np - mean_train_X) / std_train_X

                # Construct train_loader, test_loader, valid_loader is None
                train_dataset = uic_tabular_dataset(train_X, train_mask)
                test_dataset = uic_tabular_dataset(test_X, test_mask)

                train_loader = DataLoader(
                    train_dataset,
                    batch_size=config["train"]["batch_size"], 
                    shuffle=1
                )

                eval_loader_train = DataLoader(
                    train_dataset,
                    batch_size=config["train"]["batch_size"], 
                    shuffle=False
                )

                eval_loader_test = DataLoader(
                    test_dataset,
                    batch_size=config["train"]["batch_size"], 
                    shuffle=False
                )

                valid_loader = None

                model = TabCSDI(config, args.device).to(args.device)

                if args.modelfolder == "":
                    train(
                        model,
                        config["train"],
                        train_loader,
                        valid_loader=valid_loader,
                        foldername=foldername,
                    )
                else:
                    model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))
                print("---------------Start testing---------------")
                
                # In sample result
                filled_train_X = evaluate(model, eval_loader_train, nsample=args.nsample, scaler=1, foldername=foldername)
                # Out sample result
                filled_test_X = evaluate(model, eval_loader_test, nsample=args.nsample, scaler=1, foldername=foldername)
                
                # undo minmax scaling
                filled_train_X = train_dataset.minmax_scalar.inverse_transform(filled_train_X.reshape(*train_X.shape).cpu())
                filled_test_X = test_dataset.minmax_scalar.inverse_transform(filled_test_X.reshape(*test_X.shape).cpu())

                # ============In smaple evaluation================
                # apply zero mean unit var normalize 
                mean_X_filled = filled_train_X.mean(0)
                std_X_filled = filled_train_X.std(0)
                # avoid division by zero
                std_X_filled = np.where(std_X_filled == 0, 1e-3, std_X_filled)

                len_num = train_num.shape[1]
                X_filled_train_num = filled_train_X[:, :len_num].copy()
                X_filled_train_num = (X_filled_train_num - mean_X_filled[:len_num]) / std_X_filled[:len_num]
                filled_train_X[:, :len_num] = X_filled_train_num
                    
                mae_tr, rmse_tr, acc_tr = get_eval(dataname, filled_train_X, X_train_true_np, train_cat_idx, train_num.shape[1], cat_bin_num, ori_train_mask)
                
                print('Saving TabCSDI in sample results...')  
                # 1. save imputation   
                np.save(f'{ckpt_train}/TabCSDI.npy', filled_train_X) 
                # 2. write in_sample performance
                with open(f'{ckpt_in_sample}/in_sample_result_mask{mask_idx}.txt', 'w') as file: 
                    file.write(f"TabCSDI in sample MAE: {mae_tr}\n")
                    file.write(f"TabCSDI in sample RMSE: {rmse_tr}\n")
                    file.write(f"TabCSDI in sample acc: {acc_tr}\n")

                # ============Out smaple evaluation================
                # apply zero mean unit var normalize 
                mean_X_filled = filled_test_X.mean(0)
                std_X_filled = filled_test_X.std(0)
                # avoid division by zero
                std_X_filled = np.where(std_X_filled == 0, 1e-3, std_X_filled)

                len_num = test_num.shape[1]
                X_filled_test_num = filled_test_X[:, :len_num].copy()
                X_filled_test_num = (X_filled_test_num - mean_X_filled[:len_num]) / std_X_filled[:len_num]
                filled_test_X[:, :len_num] = X_filled_test_num
                    
                mae_te, rmse_te, acc_te = get_eval(dataname, filled_test_X, X_test_true_np, test_cat_idx, test_num.shape[1], cat_bin_num, ori_test_mask)
                
                print('Saving TabCSDI in sample results...')  
                # 1. save imputation   
                np.save(f'{ckpt_test}/TabCSDI.npy', filled_test_X) 
                # 2. write in_sample performance
                with open(f'{ckpt_out_sample}/out_sample_result_mask{mask_idx}.txt', 'w') as file:  #Typo: in_sample -> out_sample
                    file.write(f"TabCSDI out sample MAE: {mae_te}\n")
                    file.write(f"TabCSDI out sample RMSE: {rmse_te}\n")
                    file.write(f"TabCSDI out sample acc: {acc_te}\n")

                


                
                

                


                
                
