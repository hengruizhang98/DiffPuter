import numpy as np
import pandas as pd
import torch

from scipy import optimize

import os
import json

import argparse

from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

import pdb

DATA_DIR = 'datasets'
torch.set_default_dtype(torch.float32)

'''
Load dataset, the category data is replaced by index of the category

Note: 
the returned data's column order is the same as the original data column order 

'''
def load_dataset(dataname, idx = 0):
    data_dir = f'{DATA_DIR}/{dataname}'

    info_path = f'{DATA_DIR}/Info/{dataname}.json'

    with open(info_path, 'r') as f:
        info = json.load(f)

    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    data_path = f'{data_dir}/data.csv'
    train_path = f'{data_dir}/train.csv'
    test_path = f'{data_dir}/test.csv'

    data_df = pd.read_csv(data_path)
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    cols = train_df.columns

    #data_num = data_df[cols[num_col_idx]].values.astype(np.float32)
    data_cat = data_df[cols[cat_col_idx]].astype(str)
    data_y = data_df[cols[target_col_idx]]

    train_num = train_df[cols[num_col_idx]].values.astype(np.float32)
    train_cat = train_df[cols[cat_col_idx]].astype(str)
    train_y = train_df[cols[target_col_idx]]

    test_num = test_df[cols[num_col_idx]].values.astype(np.float32)
    test_cat = test_df[cols[cat_col_idx]].astype(str)
    test_y = test_df[cols[target_col_idx]]
    
    cat_columns = data_cat.columns
    target_columns = data_y.columns

    train_cat_idx, test_cat_idx = None, None

    # Save target idx for target columns
    if len(target_col_idx) != 0 and not is_numeric_dtype(data_y[target_columns[0]]): 
        if not os.path.exists(f'{data_dir}/{target_columns[0]}_map_idx.json'):
            print('Creating maps')
            for column in target_columns:
                map_path_bin = f'{data_dir}/{column}_map_bin.json'
                map_path_idx = f'{data_dir}/{column}_map_idx.json'
                categories = data_y[column].unique()
                num_categories = len(categories) 

                num_bits = (num_categories - 1).bit_length()

                category_to_binary = {category: format(index, '0' + str(num_bits) + 'b') for index, category in enumerate(categories)}
                category_to_idx = {category: index for index, category in enumerate(categories)}
                
                with open(map_path_bin, 'w') as f:
                    json.dump(category_to_binary, f)
                with open(map_path_idx, 'w') as f:
                    json.dump(category_to_idx, f) 
    
        train_target_idx = []
        test_target_idx = []
                
        for column in target_columns:
            map_path_idx = f'{data_dir}/{column}_map_idx.json'
            
            with open(map_path_idx, 'r') as f:
                category_to_idx = json.load(f)
                
            train_target_idx_i = train_y[column].map(category_to_idx).to_numpy().astype(np.float32)
            test_target_idx_i = test_y[column].map(category_to_idx).to_numpy().astype(np.float32)
            
            train_target_idx.append(train_target_idx_i)
            test_target_idx.append(test_target_idx_i)
        
        train_target_idx = np.stack(train_target_idx, axis = 1)
        test_target_idx = np.stack(test_target_idx, axis = 1)
    
    else:
        #abuse notation, if the target column is numeric, we still use call it target_idx
        train_target_idx = train_y.to_numpy().astype(np.float32)
        test_target_idx = test_y.to_numpy().astype(np.float32)
    
    # ========================================================

    # Save cat idx for cat columns
    if len(cat_col_idx) != 0 and not os.path.exists(f'{data_dir}/{cat_columns[0]}_map_idx.json'):
        print('Creating maps')
        for column in cat_columns:
            map_path_bin = f'{data_dir}/{column}_map_bin.json'
            map_path_idx = f'{data_dir}/{column}_map_idx.json'
            categories = data_cat[column].unique()
            num_categories = len(categories) 

            num_bits = (num_categories - 1).bit_length()

            category_to_binary = {category: format(index, '0' + str(num_bits) + 'b') for index, category in enumerate(categories)}
            category_to_idx = {category: index for index, category in enumerate(categories)}
            
            with open(map_path_bin, 'w') as f:
                json.dump(category_to_binary, f)
            with open(map_path_idx, 'w') as f:
                json.dump(category_to_idx, f)
    
            
    train_cat_idx = []
    test_cat_idx = []
            
    for column in cat_columns:
        map_path_idx = f'{data_dir}/{column}_map_idx.json'
        
        with open(map_path_idx, 'r') as f:
            category_to_idx = json.load(f)
            
        train_cat_idx_i = train_cat[column].map(category_to_idx).to_numpy().astype(np.float32)
        test_cat_idx_i = test_cat[column].map(category_to_idx).to_numpy().astype(np.float32)
        
        train_cat_idx.append(train_cat_idx_i)
        test_cat_idx.append(test_cat_idx_i)

    # Four situations:
    # 1. No target columns, no cat columns
    # 2. No target columns, has cat columns
    # 3. Has target columns, no cat columns
    # 4. Has target columns, has cat columns
    if len(target_col_idx) == 0:

        if len(cat_col_idx) == 0:
            train_X = train_num
            test_X = test_num
            
            #rearange the column order
            train_X = train_X[:, num_col_idx]
            test_X = test_X[:, num_col_idx]
        else:
            train_cat_idx = np.stack(train_cat_idx, axis = 1)
            test_cat_idx = np.stack(test_cat_idx, axis = 1)

            train_X = np.concatenate([train_num, train_cat_idx], axis = 1)
            test_X = np.concatenate([test_num, test_cat_idx], axis = 1)

            #rearange the column order
            train_X = train_X[:, np.concatenate([num_col_idx, cat_col_idx])]
            test_X = test_X[:, np.concatenate([num_col_idx, cat_col_idx])]
    
    else:
        if len(cat_col_idx) == 0:
            train_X = np.concatenate([train_num, train_target_idx], axis = 1)
            test_X = np.concatenate([test_num, test_target_idx], axis = 1)

            #rearange the column order
            train_X = train_X[:, np.concatenate([num_col_idx, target_col_idx])]
            test_X = test_X[:, np.concatenate([num_col_idx, target_col_idx])]
            
        else:
            train_cat_idx = np.stack(train_cat_idx, axis = 1)
            test_cat_idx = np.stack(test_cat_idx, axis = 1)
            
            train_X = np.concatenate([train_num, train_cat_idx, train_target_idx], axis = 1)
            test_X = np.concatenate([test_num, test_cat_idx, test_target_idx], axis = 1)

            #rearange the column order
            train_X = train_X[:, np.concatenate([num_col_idx, cat_col_idx, target_col_idx])]
            test_X = test_X[:, np.concatenate([num_col_idx, cat_col_idx, target_col_idx])]
        
    return train_X, test_X
    

#### Quantile ######
def quantile(X, q, dim=None):
    """
    Returns the q-th quantile.

    Parameters
    ----------
    X : torch.DoubleTensor or torch.cuda.DoubleTensor, shape (n, d)
        Input data.

    q : float
        Quantile level (starting from lower values).

    dim : int or None, default = None
        Dimension allong which to compute quantiles. If None, the tensor is flattened and one value is returned.


    Returns
    -------
        quantiles : torch.DoubleTensor

    """
    return X.kthvalue(int(q * len(X)), dim=dim)[0]

    

##################### MISSING DATA MECHANISMS #############################

##### Missing At Random ######

def MAR_mask(X, p, p_obs):
    """
    Missing at random mechanism with a logistic masking model. First, a subset of variables with *no* missing values is
    randomly selected. The remaining variables have missing values according to a logistic model with random weights,
    re-scaled so as to attain the desired proportion of missing values on those variables.

    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated. If a numpy array is provided,
        it will be converted to a pytorch tensor.

    p : float
        Proportion of missing values to generate for variables which will have missing values.

    p_obs : float
        Proportion of variables with *no* missing values that will be used for the logistic masking model.

    Returns
    -------
    mask : torch.BoolTensor or np.ndarray (depending on type of X)
        Mask of generated missing values (True if the value is missing).

    """

    n, d = X.shape

    to_torch = torch.is_tensor(X) ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = torch.from_numpy(X)

    mask = torch.zeros(n, d).bool() if to_torch else np.zeros((n, d)).astype(bool)

    d_obs = max(int(p_obs * d), 1) ## number of variables that will have no missing values (at least one variable)
    d_na = d - d_obs ## number of variables that will have missing values

    ### Sample variables that will all be observed, and those with missing values:
    idxs_obs = np.random.choice(d, d_obs, replace=False)
    idxs_nas = np.array([i for i in range(d) if i not in idxs_obs])

    ### Other variables will have NA proportions that depend on those observed variables, through a logistic model
    ### The parameters of this logistic model are random.

    ### Pick coefficients so that W^Tx has unit variance (avoids shrinking)
    coeffs = pick_coeffs(X, idxs_obs, idxs_nas)
    ### Pick the intercepts to have a desired amount of missing values
    intercepts = fit_intercepts(X[:, idxs_obs], coeffs, p)

    ps = torch.sigmoid(X[:, idxs_obs].mm(coeffs) + intercepts)

    ber = torch.rand(n, d_na)
    mask[:, idxs_nas] = ber < ps

    return mask

##### Missing not at random ######

def MNAR_mask_logistic(X, p, p_params =.3, exclude_inputs=True):
    """
    Missing not at random mechanism with a logistic masking model. It implements two mechanisms:
    (i) Missing probabilities are selected with a logistic model, taking all variables as inputs. Hence, values that are
    inputs can also be missing.
    (ii) Variables are split into a set of intputs for a logistic model, and a set whose missing probabilities are
    determined by the logistic model. Then inputs are then masked MCAR (hence, missing values from the second set will
    depend on masked values.
    In either case, weights are random and the intercept is selected to attain the desired proportion of missing values.

    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
        If a numpy array is provided, it will be converted to a pytorch tensor.

    p : float
        Proportion of missing values to generate for variables which will have missing values.

    p_params : float
        Proportion of variables that will be used for the logistic masking model (only if exclude_inputs).

    exclude_inputs : boolean, default=True
        True: mechanism (ii) is used, False: (i)

    Returns
    -------
    mask : torch.BoolTensor or np.ndarray (depending on type of X)
        Mask of generated missing values (True if the value is missing).

    """

    n, d = X.shape

    to_torch = torch.is_tensor(X) ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = torch.from_numpy(X)

    mask = torch.zeros(n, d).bool() if to_torch else np.zeros((n, d)).astype(bool)

    d_params = max(int(p_params * d), 1) if exclude_inputs else d ## number of variables used as inputs (at least 1)
    d_na = d - d_params if exclude_inputs else d ## number of variables masked with the logistic model

    ### Sample variables that will be parameters for the logistic regression:
    idxs_params = np.random.choice(d, d_params, replace=False) if exclude_inputs else np.arange(d)
    idxs_nas = np.array([i for i in range(d) if i not in idxs_params]) if exclude_inputs else np.arange(d)

    ### Other variables will have NA proportions selected by a logistic model
    ### The parameters of this logistic model are random.

    ### Pick coefficients so that W^Tx has unit variance (avoids shrinking)
    coeffs = pick_coeffs(X, idxs_params, idxs_nas)
    ### Pick the intercepts to have a desired amount of missing values
    intercepts = fit_intercepts(X[:, idxs_params], coeffs, p)

    ps = torch.sigmoid(X[:, idxs_params].mm(coeffs) + intercepts)

    ber = torch.rand(n, d_na)
    mask[:, idxs_nas] = ber < ps

    ## If the inputs of the logistic model are excluded from MNAR missingness,
    ## mask some values used in the logistic model at random.
    ## This makes the missingness of other variables potentially dependent on masked values

    if exclude_inputs:
        mask[:, idxs_params] = torch.rand(n, d_params) < p

    return mask

def MNAR_self_mask_logistic(X, p):
    """
    Missing not at random mechanism with a logistic self-masking model. Variables have missing values probabilities
    given by a logistic model, taking the same variable as input (hence, missingness is independent from one variable
    to another). The intercepts are selected to attain the desired missing rate.

    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
        If a numpy array is provided, it will be converted to a pytorch tensor.

    p : float
        Proportion of missing values to generate for variables which will have missing values.

    Returns
    -------
    mask : torch.BoolTensor or np.ndarray (depending on type of X)
        Mask of generated missing values (True if the value is missing).

    """

    n, d = X.shape

    to_torch = torch.is_tensor(X) ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = torch.from_numpy(X)

    ### Variables will have NA proportions that depend on those observed variables, through a logistic model
    ### The parameters of this logistic model are random.

    ### Pick coefficients so that W^Tx has unit variance (avoids shrinking)
    coeffs = pick_coeffs(X, self_mask=True)
    ### Pick the intercepts to have a desired amount of missing values
    intercepts = fit_intercepts(X, coeffs, p, self_mask=True)

    ps = torch.sigmoid(X * coeffs + intercepts)

    ber = torch.rand(n, d) if to_torch else np.random.rand(n, d)
    mask = ber < ps if to_torch else ber < ps.numpy()

    return mask


def MNAR_mask_quantiles(X, p, q, p_params, cut='both', MCAR=False):
    """
    Missing not at random mechanism with quantile censorship. First, a subset of variables which will have missing
    variables is randomly selected. Then, missing values are generated on the q-quantiles at random. Since
    missingness depends on quantile information, it depends on masked values, hence this is a MNAR mechanism.

    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
        If a numpy array is provided, it will be converted to a pytorch tensor.

    p : float
        Proportion of missing values to generate for variables which will have missing values.

    q : float
        Quantile level at which the cuts should occur

    p_params : float
        Proportion of variables that will have missing values

    cut : 'both', 'upper' or 'lower', default = 'both'
        Where the cut should be applied. For instance, if q=0.25 and cut='upper', then missing values will be generated
        in the upper quartiles of selected variables.
        
    MCAR : bool, default = True
        If true, masks variables that were not selected for quantile censorship with a MCAR mechanism.
        
    Returns
    -------
    mask : torch.BoolTensor or np.ndarray (depending on type of X)
        Mask of generated missing values (True if the value is missing).

    """
    n, d = X.shape

    to_torch = torch.is_tensor(X) ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = torch.from_numpy(X)

    mask = torch.zeros(n, d).bool() if to_torch else np.zeros((n, d)).astype(bool)

    d_na = max(int(p_params * d), 1) ## number of variables that will have NMAR values

    ### Sample variables that will have imps at the extremes
    idxs_na = np.random.choice(d, d_na, replace=False) ### select at least one variable with missing values

    ### check if values are greater/smaller that corresponding quantiles
    if cut == 'upper':
        quants = quantile(X[:, idxs_na], 1-q, dim=0)
        m = X[:, idxs_na] >= quants
    elif cut == 'lower':
        quants = quantile(X[:, idxs_na], q, dim=0)
        m = X[:, idxs_na] <= quants
    elif cut == 'both':
        u_quants = quantile(X[:, idxs_na], 1-q, dim=0)
        l_quants = quantile(X[:, idxs_na], q, dim=0)
        m = (X[:, idxs_na] <= l_quants) | (X[:, idxs_na] >= u_quants)

    ### Hide some values exceeding quantiles
    ber = torch.rand(n, d_na)
    mask[:, idxs_na] = (ber < p) & m

    if MCAR:
    ## Add a mcar mecanism on top
        mask = mask | (torch.rand(n, d) < p)

    return mask


def pick_coeffs(X, idxs_obs=None, idxs_nas=None, self_mask=False):
    n, d = X.shape
    if self_mask:
        coeffs = torch.randn(d)
        Wx = X * coeffs
        coeffs /= torch.std(Wx, 0)
    else:
        d_obs = len(idxs_obs)
        d_na = len(idxs_nas)
        coeffs = torch.randn(d_obs, d_na, dtype=X.dtype)
        Wx = X[:, idxs_obs].mm(coeffs)
        coeffs /= torch.std(Wx, 0, keepdim=True)
    return coeffs


def fit_intercepts(X, coeffs, p, self_mask=False):
    if self_mask:
        d = len(coeffs)
        intercepts = torch.zeros(d)
        for j in range(d):
            def f(x):
                return torch.sigmoid(X * coeffs[j] + x).mean().item() - p
            intercepts[j] = optimize.bisect(f, -50, 50)
    else:
        d_obs, d_na = coeffs.shape
        intercepts = torch.zeros(d_na)
        for j in range(d_na):
            def f(x):
                return torch.sigmoid(X.mv(coeffs[:, j]) + x).mean().item() - p
            left,right = -500, 500
            intercepts[j] = optimize.bisect(f, left, right)
    return intercepts


def generate_mask(dataname, mask_type, p, mask_num, reproduce=True):


    train_X, test_X = load_dataset(dataname)
    print('missing probability:', p)

    #p = p / (1 - 0.3) # 30% will held out and not missing, so we need to adjust the missing probability 
    q = 0.3
    if p > 0.3:
        q = 0.1
    
    for mask_idx in range(mask_num):
        if mask_type == 'MCAR':
            #train_mask = (torch.rand(train_X.shape) < p).numpy()
            #test_mask = (torch.rand(test_X.shape) < p).numpy()
            train_mask = np.random.rand(*train_X.shape) < p
            test_mask = np.random.rand(*test_X.shape) < p
        elif mask_type == 'MAR':
            train_mask = MAR_mask(train_X, p=p/(1-q), p_obs=q)
            test_mask = MAR_mask(test_X, p=p/(1-q), p_obs=q)
        elif mask_type == 'MNAR_logistic_T2':
            train_mask = MNAR_mask_logistic(train_X, p=p, p_params=q, exclude_inputs=True)
            test_mask = MNAR_mask_logistic(test_X, p=p, p_params=q, exclude_inputs=True)
        # elif mask_type == 'MNAR_logistic_T1':
        #     train_mask = MNAR_mask_logistic(train_X, p=0.3, p_obs=0.3, exclude_inputs=False)
        #     test_mask = MNAR_mask_logistic(test_X, p=0.3, p_obs=0.3, exclude_inputs=False)
        # elif mask_type == 'MNAR_self_logistic':
        #     train_mask = MNAR_self_mask_logistic(train_X, p=0.3)
        #     test_mask = MNAR_self_mask_logistic(test_X, p=0.3)
        # elif mask_type == 'MNAR_quantiles':
        #     train_mask = MNAR_mask_quantiles(train_X, 0.3, 0.25, 0.3, cut='both', MCAR=False)
        #     test_mask = MNAR_mask_quantiles(test_X, 0.3, 0.25, 0.3, cut='both', MCAR=False)
        else:
            raise ValueError('Invalid mask type, please choose from MCAR, MAR, MNAR_logistic_T2')
        
        row, col = train_mask.shape
        print('train_mask missing prob:', np.sum(train_mask) / (row * col))
    
        folder_name = 'rate' + str(int(p * 100))
      
        data_dir = f'{DATA_DIR}/{dataname}/masks/{folder_name}/{mask_type}'
        
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        train_mask_path = f'{data_dir}/train_mask_{mask_idx}.npy'
        test_mask_path = f'{data_dir}/test_mask_{mask_idx}.npy'
        
        # If exist, pass
        if os.path.exists(train_mask_path) and os.path.exists(test_mask_path) and not reproduce:
            print(f'Masks already exist at {train_mask_path}')
            continue
        
        # Save train/test masks
        np.save(train_mask_path, train_mask)
        np.save(test_mask_path, test_mask)

        print(f'Saved train mask to {train_mask_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MAR and MNAR mask generation.')

    parser.add_argument('--dataname', type=str, default='california', help='Name of dataset.')
    parser.add_argument('--mask_type', type=str, default='MNAR_logistic_T2', help='Type of missing data mechanism.')
    parser.add_argument('--mask_num', type=int, default=10)
    parser.add_argument('--p', type=float, default=0.3, help='Proportion of missing values.')
    parser.add_argument('-reproduce', action='store_true', help='Reproduce masks.')

    args = parser.parse_args()

    generate_mask(args.dataname, args.mask_type, args.mask_num, args.p, args.reproduce)