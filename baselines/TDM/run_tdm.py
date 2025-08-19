import numpy as np
import os
import torch
import logging
from .tdm import TDM, TDM_RRimputer
import ot
from .utils import MAE, RMSE, MLP_Net
import torch.nn as nn

import FrEIA.framework as Ff
import FrEIA.modules as Fm

import pdb

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
else:
    torch.set_default_tensor_type('torch.DoubleTensor')


def run_TDM(X_missing, args, X_true=None, imputer='TDM', return_imputer=False, rr_model='linear'):
    '''
    Parameters
    ----------
    X_missing : np.ndarray
        The input data with missing values.

    args : dict
        The arguments for the imputer.

    X_true : np.ndarray, optional   
        The ground truth data.

    imputer : str, optional 
        The imputer to use. 
        Options are 'TDM' and 'RR_TDM'.     

    return_imputer : bool, optional
        Whether to return the imputer.  

    rr_model : str, optional        
        The model to use for RR_TDM.
        Options are 'linear' and 'mlp'.   

    Returns      
    ------- 
    result : dict
        The imputed data and the evaluation metrics.

    imputer : TDM or TDM_RRimputer
    '''

    save_dir = args['out_dir']
    # if save_dir is not None:
    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir)

    #FORMAT = '%(asctime)-15s %(message)s'
    #logging.basicConfig(level=logging.DEBUG, format=FORMAT, filename=os.path.join(save_dir, 'log.txt'))

    # For small datasets, smaller batchsize may prevent overfitting; 
    # For larger datasets, larger batchsize may give better performance.
    if 'batchsize' in args: 
        batchsize = args['batchsize']
    else:
        batchsize = 512

    X_missing = torch.Tensor(X_missing)
    if X_true is not None:
        X_true = torch.Tensor(X_true)
    n, d = X_missing.shape
    mask = torch.isnan(X_missing)

    k = args['network_width']
    def subnet_fc(dims_in, dims_out):
        return nn.Sequential(nn.Linear(dims_in, k * d), nn.SELU(),  nn.Linear(k * d, k * d), nn.SELU(),
                            nn.Linear(k * d,  dims_out))
    projector = Ff.SequenceINN(d)
    for _ in range(args['network_depth']):
        projector.append(Fm.RNVPCouplingBlock, subnet_constructor=subnet_fc)
    
    if imputer == 'TDM':
        imputer = TDM(projector,
                      batchsize=batchsize, 
                      im_lr=args['lr'], 
                      proj_lr=args['lr'], 
                      niter=args['niter'], 
                      save_dir_training=save_dir)
        
    elif imputer == 'RR_TDM':
        if rr_model == 'linear':
            #Linear model 
            d_ = d - 1
            models = {}

            for i in range(d):
                models[i] = nn.Linear(d_, 1)
        
        elif rr_model == 'mlp':
            #MLP model
            d_ = d - 1
            models = {}
          
            for i in range(d):
                # models[i] = nn.Sequential(nn.Linear(d_, 2 * d_),
                #                         nn.ReLU(),
                #                         nn.Linear(2 * d_, d_),
                #                         nn.ReLU(),
                #                         nn.Linear(d_, 1))
                
                #https://discuss.pytorch.org/t/runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation-torch-floattensor-64-1-which-is-output-0-of-asstridedbackward0-is-at-version-3-expected-version-2-instead-hint-the-backtrace-further-a/171826/7
                #a workaround to avoid the inplace operation error for retain_graph=True
                models[i] = MLP_Net(d_) 
        
        else:
            raise NotImplementedError(f"{rr_model} not exist. Please choose between 'linear' and 'mlp'.")
            
        imputer = TDM_RRimputer(models,
                                projector,
                                batchsize=batchsize,
                                rr_lr=args['rr_lr'],
                                proj_lr=args['proj_lr'],
                                niter=args['niter'],
                                max_niter=args['max_niter'],
                                n_pairs=args['n_pairs'],
                                save_dir_training=save_dir,
                                )
    else:
        raise NotImplementedError(f"{imputer} not exist. Please choose between 'TDM' and 'RR_TDM'.")
    
    imp, maes, rmses = imputer.fit_transform(X_missing.clone(), verbose=True, report_interval=args['report_interval'], X_true=X_true)
    imp = imp.detach()

    result = {}
    result["imp"] = imp[mask.bool()].detach().cpu().numpy()
    result['filled_X'] = imp.detach().cpu().numpy()
    
    if X_true is not None:
        result['learning_MAEs'] = maes
        result['learning_RMSEs'] = rmses
        result['MAE'] = MAE(imp, X_true, mask).item()
        result['RMSE'] = RMSE(imp, X_true, mask).item()
        OTLIM = 5000
        M = mask.sum(1) > 0
        nimp = M.sum().item()
        if nimp < OTLIM:
            M = mask.sum(1) > 0
            nimp = M.sum().item()
            dists = ((imp[M][:, None] - X_true[M]) ** 2).sum(2) / 2.
            result['OT'] = ot.emd2(np.ones(nimp) / nimp,
                                        np.ones(nimp) / nimp, \
                                        dists.cpu().numpy())
            logging.info(
                    f"MAE: {result['MAE']:.4f}\t"
                    f"RMSE: {result['RMSE']:.4f}\t"
                    f"OT: {result['OT']:.4f}")
        else:
            logging.info(
                    f"MAE: {result['MAE']:.4f}\t"
                    f"RMSE: {result['RMSE']:.4f}\t")
            
    #np.save(os.path.join(save_dir, 'result.npy'), result)

    if return_imputer:
        return result, imputer
    else:
        return result
