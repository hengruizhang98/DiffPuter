import os
import numpy as np
import pandas as pd
from urllib import request
import shutil
import zipfile
import json
from generate_mask import generate_mask

DATA_DIR = 'datasets'


NAME_URL_DICT_UCI = {
    'adult': 'https://archive.ics.uci.edu/static/public/2/adult.zip',
    'default': 'https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip',
    'magic': 'https://archive.ics.uci.edu/static/public/159/magic+gamma+telescope.zip',
    'shoppers': 'https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip',
    'beijing': 'https://archive.ics.uci.edu/static/public/381/beijing+pm2+5+data.zip',
    'news': 'https://archive.ics.uci.edu/static/public/332/online+news+popularity.zip',
    'gesture': 'https://archive.ics.uci.edu/static/public/302/gesture+phase+segmentation.zip',
    'letter': 'https://archive.ics.uci.edu/static/public/59/letter+recognition.zip',
    'bean': 'https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip'
}

def unzip_file(zip_filepath, dest_path):
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        zip_ref.extractall(dest_path)


def download_from_uci(name):

    print(f'Start processing dataset {name} from UCI.')
    save_dir = f'{DATA_DIR}/{name}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

        url = NAME_URL_DICT_UCI[name]
        request.urlretrieve(url, f'{save_dir}/{name}.zip')
        print(f'Finish downloading dataset from {url}, data has been saved to {save_dir}.')
        
        unzip_file(f'{save_dir}/{name}.zip', save_dir)
        print(f'Finish unzipping {name}.')
    
    else:
        print('Aready downloaded.')

def process_adult():
    data_dir = f'{DATA_DIR}/adult'
    df = pd.read_csv(f'{data_dir}/adult.data', header=None)
    df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', ]
                  
def process_gesture():

    file_names = ['a1_va3', 'a2_va3', 'a3_va3', 'b1_va3', 'b1_va3', 'c1_va3', 'c3_va3']
    datas = []
    for name in file_names:
        df = pd.read_csv(f'{DATA_DIR}/gesture/{name}.csv')
        data = df.to_numpy()
        datas.append(data)

    data = np.concatenate(datas, axis=0)
    data_df = pd.DataFrame(data)
    data_df.to_csv(f'{DATA_DIR}/gesture/data.csv', index = False)


def process_letter():
    dataname = 'letter'
    path = f'{DATA_DIR}/{dataname}/{dataname}-recognition.data'
    save_path = f'{DATA_DIR}/{dataname}/data.csv'
    df = pd.read_csv(path, header = None)

    cols = df.columns.tolist()
    cols = cols[1:] + cols[:1]

    df = df[cols]
    df.to_csv(save_path, index = False, header = True)

def process_news():
    path = f'{DATA_DIR}/news/OnlineNewsPopularity/OnlineNewsPopularity.csv'
    save_path = f'{DATA_DIR}/news/data.csv'

    data_df = pd.read_csv(path)
    data_df = data_df.drop('url', axis=1)

    columns = np.array(data_df.columns.tolist())

    cat_columns1 = columns[list(range(12,18))]
    cat_columns2 = columns[list(range(30,38))]

    cat_col1 = data_df[cat_columns1].astype(int).to_numpy().argmax(axis = 1)
    cat_col2 = data_df[cat_columns2].astype(int).to_numpy().argmax(axis = 1)

    data_df = data_df.drop(cat_columns2, axis=1)
    data_df = data_df.drop(cat_columns1, axis=1)

    data_df['data_channel'] = cat_col1
    data_df['weekday'] = cat_col2
    
    data_df.to_csv(f'{save_path}', index = False)

def process_beijing():
    path = f'{DATA_DIR}/beijing/PRSA_data_2010.1.1-2014.12.31.csv'
    save_path = f'{DATA_DIR}/beijing/data.csv'
    data_df = pd.read_csv(path)
    data_df = data_df.drop('No', axis=1)

    df_cleaned = data_df.dropna()
    df_cleaned.to_csv(save_path, index = False)

def process_adult():
    path = f'{DATA_DIR}/adult/adult.data'
    save_path = f'{DATA_DIR}/adult/data.csv'
    data_df = pd.read_csv(path, header=None)

    df_cleaned = data_df.dropna()
    df_cleaned.to_csv(save_path, index = False)



def process_shoppers():
    path = f'{DATA_DIR}/shoppers/online_shoppers_intention.csv'
    save_path = f'{DATA_DIR}/shoppers/data.csv'
    data_df = pd.read_csv(path)

    df_cleaned = data_df.dropna()
    df_cleaned.to_csv(save_path, index = False)

def process_default():
    path = f'{DATA_DIR}/default/default of credit card clients.xls'
    save_path = f'{DATA_DIR}/default/data.csv'
    data_df = pd.read_excel(path, sheet_name='Data', header=1)
    data_df = data_df.drop('ID', axis=1)

    df_cleaned = data_df.dropna()
    df_cleaned.to_csv(save_path, index = False)

def process_magic():

    path = f'{DATA_DIR}/magic/magic04.data'
    save_path = f'{DATA_DIR}/magic/data.csv'
    data_df = pd.read_csv(path, header=None)
    columns = data_df.columns
    df_cleaned = data_df.dropna()
    df_cleaned.to_csv(save_path, index = False)

def process_bean():

    path = f'{DATA_DIR}/bean/DryBeanDataset/Dry_Bean_Dataset.xlsx'
    save_path = f'{DATA_DIR}/bean/data.csv'
    data_df = pd.read_excel(path, sheet_name='Dry_Beans_Dataset', header=1)

    df_cleaned = data_df.dropna()
    df_cleaned.to_csv(save_path, index = False)

def train_test_split(dataname, ratio = 0.7, mask_prob = 0.3):
    data_dir = f'{DATA_DIR}/{dataname}'
    path = f'{DATA_DIR}/{dataname}/data.csv'
    info_path = f'{DATA_DIR}/Info/{dataname}.json'

    with open(info_path, 'r') as f:
        info = json.load(f)

    cat_idx = info['cat_col_idx']
    num_idx = info['num_col_idx']

    data_df = pd.read_csv(path)
    total_num = data_df.shape[0]

    if len(cat_idx) == 0:
        data_values = data_df.values[:, :-1].astype(np.float32)

        nan_idx = np.isnan(data_values).nonzero()[0]

        keep_idx = list(set(np.arange(data_values.shape[0])) - set(list(nan_idx)))
        keep_idx = np.array(keep_idx)
    else:
        keep_idx = np.arange(total_num)

    num_train = int(keep_idx.shape[0] * ratio)
    num_test = total_num - num_train
    seed = 1234

    np.random.seed(seed)
    np.random.shuffle(keep_idx)

    train_idx = keep_idx[:num_train]
    test_idx = keep_idx[-num_test:]

    train_df = data_df.loc[train_idx]
    test_df = data_df.loc[test_idx]        

    train_path = f'{data_dir}/train.csv'
    test_path = f'{data_dir}/test.csv'

    train_df.to_csv(train_path, index = False)
    test_df.to_csv(test_path, index = False)

    print(f'Spliting Trainig and Testing data for {dataname} is done.')
    print(f'Training data shape: {train_df.shape}, Testing data shape: {test_df.shape}')
    print(f'Training data saved at {train_path}, Testing data saved at {test_path}.')
    

    # train_X = train_df.to_numpy()
    # test_X = test_df.to_numpy()
    
    # X_train = train_X
    # X_test = test_X

    # os.makedirs(f'{data_dir}/masks/MCAR') if not os.path.exists(f'{data_dir}/masks/MCAR') else None
    # for cross_idx in range(10):
    #     np.random.seed(cross_idx) 

    #     mask_train = np.random.rand(*X_train.shape) < mask_prob
    #     mask_test = np.random.rand(*X_test.shape) < mask_prob
        
    #     np.save(f'{data_dir}/masks/MCAR/train_mask_{cross_idx}.npy', mask_train)
    #     np.save(f'{data_dir}/masks/MCAR/test_mask_{cross_idx}.npy', mask_test)

    # print(dataname, mask_train.shape, train_X.shape, test_X.shape, len(num_idx), len(cat_idx))
    

if __name__ == '__main__':

    # Downloading dataset
    for name in NAME_URL_DICT_UCI.keys():
        download_from_uci(name)

    for name in NAME_URL_DICT_UCI.keys():
        eval(f'process_{name}()')
        train_test_split(name, ratio = 0.7, mask_prob = 0.3)
        for mask_type in ['MCAR', 'MAR', 'MNAR_logistic_T2']:
            for mask_p in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]:
                
                generate_mask(dataname = name,
                                mask_type = mask_type,
                                mask_num = 10,
                                p = mask_p,
                                )
    

