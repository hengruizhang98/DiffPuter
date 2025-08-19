import argparse
import sys
import os
import os.path as osp

import numpy as np
#from sympy import arg
import torch

from training.gnn_mdi import train_gnn_mdi
from uci.uci_subparser import add_uci_subparser
from utils.utils import auto_select_gpu
import pickle

def main():
    parser = argparse.ArgumentParser()# domain
    parser.add_argument('--model_types', type=str, default='EGSAGE_EGSAGE_EGSAGE')
    parser.add_argument('--domain', type=str, default='uci')
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
    parser.add_argument('--node_mode', type=int, default=0)  # 0: feature onehot, sample all 1; 1: all onehot
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
    parser.add_argument('--seed', type=int, default=1314)
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

    # select device
    if torch.cuda.is_available():
        cuda = auto_select_gpu()
        cuda = 0
        device = torch.device('cuda:{}'.format(cuda))
    else:
        print('Using CPU')
        device = torch.device('cpu')

    seeds = [0,1314,100,1000,2000]
    for dataset in ['concrete','housing','wine','heart','DOW30','ecommerce','diabetes','yacht']:
        args.data = dataset
        for i in range(0,5):
            seed = seeds[i]
            args.seed = seed
            np.random.seed(seed)
            torch.manual_seed(seed)
            if args.domain == 'uci':
                from uci.uci_data import load_data
                data = load_data(args)

            log_path = './{}/test/{}/{}/{}/'.format(args.domain,args.known,args.data,args.log_dir) 

            # os.makedirs(log_path)
            if not osp.exists(log_path):
                os.makedirs(log_path)

            cmd_input = 'python ' + ' '.join(sys.argv) + ' ' + str(seed) + '\n'

            train_gnn_mdi(data, args, log_path, device)
            file = open('uci/test/{}/{}/0/result.pkl'.format(args.known,args.data),'rb')
            result = pickle.load(file)
            mae = result['curves']['test_l1'][-1]
            rmse = result['curves']['test_rmse'][-1]
            file.close()
            with open(osp.join(log_path, 'cmd_input.txt'), 'a') as f:
                f.write(cmd_input)
                f.write('MAE:' + str(mae) + '\n')
                f.write('RMSE:' + str(rmse) + '\n')



if __name__ == '__main__':
    main()