
import numpy as np
import os
from data_utils import generate_mask
from run_tdm import run_TDM

data_dir = './datasets'
dataset = 'seeds'
missing_prop = 0.3
missing_type = 'MCAR' # Choosing from MAR, MNARL, MNARQ, MCAR
data = np.load(os.path.join(data_dir,  '{}.npy'.format(dataset)), allow_pickle=True).item()
X_true = data['X_true']
mask = generate_mask(X_true, missing_prop, missing_type)
X_missing = np.copy(X_true)
X_missing[mask.astype(bool)] = np.nan

niter = 10000
batchsize = 64
lr = 1e-2
report_interval = 100
network_depth = 3
network_width = 2
args = {'out_dir': f'./demo_{dataset}', 'niter': niter,
 'batchsize': batchsize, 'lr': lr, 'network_width': network_width, 'network_depth': network_depth, 'report_interval': report_interval}


run_TDM(X_missing, args, X_true)