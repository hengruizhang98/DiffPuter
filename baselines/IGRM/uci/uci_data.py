import pickle
from numpy.core.numeric import NaN
import pandas as pd
import os.path as osp
import inspect
from torch_geometric.data import Data
from sklearn import preprocessing

import torch
import random
import numpy as np
import pdb
import math

from utils.utils import get_known_mask, mask_edge
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.fpm import FPGrowth
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from uci.adj_matrix import *
# from training.GAugO_method import *
from sklearn.metrics.pairwise import cosine_similarity
from .simulate import simulate_nan

# conf, cos, each_conf, lift
rules_attr_method = 'conf'
ratio = 0.3
initial = 'random'
mask_type = 'MCAR'

def fpgrowth(data,confidence):
    data = [[data[i]] for i in range(len(data))]
    appname = "FPgrowth"
    master ="local[4]" 

    data_list=data
    conf = SparkConf().setAppName(appname).setMaster(master)  #spark               
    spark=SparkSession.builder.config(conf=conf).getOrCreate()
    data=spark.createDataFrame(data_list,["items"])
    fp = FPGrowth(minSupport=0.1, minConfidence=confidence)  # model
    fpm  = fp.fit(data)
    fpm.freqItemsets.show()     # show top 20 in terminal
    assRule=fpm.associationRules
    assRuleDf=assRule.toPandas()

    print('Association rules：\n',assRuleDf[assRuleDf['lift'] > 1])             
    spark.stop()
    return assRuleDf

def create_node(df, mode, df_y):
    if mode == 0: # onehot feature node, all 1 sample node
        nrow, ncol = df.shape
        feature_ind = np.array(range(ncol))
        feature_node = np.zeros((ncol,ncol))
        feature_node[np.arange(ncol), feature_ind] = 1 # nrow x  ncol 
        sample_node = [[1]*ncol for i in range(nrow)] # nrow x  ncol
        node = sample_node + feature_node.tolist()  # nrow x  ncol 
    elif mode == 1: # onehot sample and feature node
        nrow, ncol = df.shape
        feature_ind = np.array(range(ncol))
        feature_node = np.zeros((ncol,ncol+1))
        feature_node[np.arange(ncol), feature_ind+1] = 1
        sample_node = np.zeros((nrow,ncol+1))
        sample_node[:,0] = 1
        node = sample_node.tolist() + feature_node.tolist()
    return node
def create_edge(df):

    n_row, n_col = df.shape
    # create fully connected bidirectional graph
    edge_start = []
    edge_end = []
    for x in range(n_row):
        edge_start = edge_start + [x] * n_col # obj
        edge_end = edge_end + list(n_row+np.arange(n_col)) # att    
    edge_start_new = edge_start + edge_end
    edge_end_new = edge_end + edge_start
    return (edge_start_new, edge_end_new)

def create_edge_attr(df):
    nrow, ncol = df.shape
    edge_attr = []
    for i in range(nrow):
        for j in range(ncol):
            edge_attr.append([float(df.iloc[i,j])])
    edge_attr = edge_attr + edge_attr
    return edge_attr

def add_tag(features):
    for i in range(features.shape[1]):
        for j in range(len(features[0])):
            if(np.isnan(features[i][j])):
                features[i][j] = features[i][j]
            else:
                features[i][j] = chr(97 + i) + str(int(features[i][j]))
    return features

def cluster_features(data,dataset):
    if dataset == 'ecommerce':
        cut_col = [4,8,9]
    elif dataset == 'telescope':
        cut_col = [0,1,2,3,4,5,6,7,8,9]
    elif dataset == 'heart':
        cut_col = [0,3,4,7]
    elif dataset == 'diabetes':
        cut_col = [0]
    elif(dataset == 'power'):
        cut_col = [0,1,2,3]
    elif(dataset == 'naval'):
        feature0 = {8.206:0, 3.144:1, 5.14:2, 6.175:3, 2.088:4, 1.138:5, 9.3:6, 7.148:7, 4.161:8}
        data.iloc[:,0] = data.iloc[:,0].map(lambda x: feature0[x])
        feature11 = {0.998:0}
        data.iloc[:,11] = data.iloc[:,11].map(lambda x:feature11[x])
        cut_col = [i for i in range(2,17)]
        cut_col.remove(8)
        cut_col.remove(11)
    elif(dataset == 'housing'):
        cut_col = [0,1,2,4,5,6,7,9,10,11,12]
    elif(dataset == 'energy'):
        cut_col = [0,1,2,3,4,6]
    elif(dataset == 'wine'):
        cut_col = [i for i in range(0,11)]
    elif(dataset == 'concrete'):
        cut_col = [0,1,2,3,4,5,6,7]
    elif(dataset == 'DOW30'):
        cut_col = [i for i in range(0,12)]
    elif(dataset == 'protein'):
        cut_col = [i for i in range(0,9)]
    elif(dataset == 'kin8nm'):
        cut_col = [i for i in range(0,8)]
    elif(dataset == 'yacht'):
        cut_col = [0,1,2,3,4,5]

    max_nbs = 10
    n_jobs = 10
    for i in cut_col:
        fea_i = data[i].to_numpy().reshape(-1, 1)
        fea = fea_i[fea_i==fea_i].reshape(-1, 1)
        k_et_list = []
        for nbs in range(2,max_nbs+1):
            k_model = KMeans(n_clusters=nbs,n_jobs=n_jobs)
            k_model.fit(fea)
            k_labels=k_model.labels_
            k_et=davies_bouldin_score(fea,k_labels)
            k_et_list.append(k_et)
        k_best_nbs=k_et_list.index(min(k_et_list))+2
        k_model = KMeans(n_clusters=k_best_nbs,n_jobs=n_jobs)
        k_model.fit(fea)
        k_labels = k_model.labels_
        n = 0
        for index,j in enumerate(fea_i):
            if(np.isnan(j)):
                data[i][index] = NaN
            else:
                data[i][index] = k_labels[n]
                n = n + 1    
    return data

def get_all_rule(assRuleDf):
    for i in range(len(assRuleDf['consequent'])):
        assRuleDf['consequent'][i] = assRuleDf['consequent'][i][0]
    df_rules_other = assRuleDf[(assRuleDf['lift'] > 1)]
    rules_other_ant = df_rules_other['antecedent'].tolist()
    rules_other_con = df_rules_other['consequent'].tolist()
    rules_other_confidence = df_rules_other['confidence'].tolist()
    rules_other_lift = df_rules_other['lift'].tolist()
    rules_other = list(zip(rules_other_ant, rules_other_con))
    j = 0
    for i in rules_other:
        res = [i for i in i[0]]
        res.append(i[1])
        rules_other[j] = res
        j += 1
    return rules_other, rules_other_confidence, rules_other_lift

def get_rules_edge_with_mask(rules_other, features):
    add_edge_start = []
    add_edge_end = []
    for rule_index, rule in enumerate(rules_other):
        index = []
        for i in range(features.shape[0]):
            # 判断一条含缺失值的sample完全包含关联规则rule
            res = [False for a in rule if a not in features[i]]
            if res==[]:
                index.append(i)
        length = np.floor(len(index) * ratio)
        length = length if length % 2==0 else length - 1
        for a in range(int(length/2)):
            edge = random.sample(index,2)
            add_edge_start.append(edge[0])
            add_edge_end.append(edge[1])
            index.remove(edge[0])
            index.remove(edge[1])
    edges_repeat = list(zip(add_edge_start, add_edge_end))
    edges_no_repeat = list(set(edges_repeat))
    add_edge_start = list(list(zip(*edges_no_repeat))[0])
    add_edge_end = list(list(zip(*edges_no_repeat))[1])
    return add_edge_start, add_edge_end

def get_cos(index0, index1, mask, features):
    mask_list = []
    for i in range(len(mask[0])):
        if mask[index0][i] == True and mask[index1][i] == True:
            mask_list.append(True)
        else:
            mask_list.append(False)
    features0 = features[index0][mask_list]
    features1 = features[index1][mask_list]
    from scipy.spatial.distance import pdist
    cos = pdist(np.vstack([features0,features1]),'cosine')
    return 1-cos

def get_edge_with_random(x, confidence, num_edge = None):
    # 对于NxN 的图 随机 generate N 那么多的边
    # 
    add_edge_start = []
    add_edge_end = []
    index = [i for i in range(x.shape[0])]  
    edge_index = []
    while(len(edge_index) < num_edge*2):
        edge = random.sample(index,2)
        if edge in edge_index:
            continue
        else:
            add_edge_start.append(edge[0])
            add_edge_end.append(edge[1])
            edge_index.append(edge)        
            edge_index.append([edge[1], edge[0]])
    return add_edge_start, add_edge_end

def get_data(df_X, df_y, node_mode, train_edge_prob, split_sample_ratio, split_by, train_y_prob, confidence, seed=0, dataset=None, normalize=True):
    if len(df_y.shape)==1:
        df_y = df_y.to_numpy()
    elif len(df_y.shape)==2:
        df_y = df_y[0].to_numpy()
    features = pd.DataFrame(df_X.values)

    if normalize:
        x = df_X.values
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df_X = pd.DataFrame(x_scaled)

    # created fully connected grpah
    edge_start, edge_end = create_edge(df_X)
    edge_index = torch.tensor([edge_start, edge_end], dtype=int)
    edge_attr = torch.tensor(create_edge_attr(df_X), dtype=torch.float)
    node_init = create_node(df_X, node_mode, df_y) 
    x = torch.tensor(node_init, dtype=torch.float)
    y = torch.tensor(df_y, dtype=torch.float)
    
    #set seed to fix known/unknwon edges
    torch.manual_seed(seed)
    # select missing mechanism
    if mask_type == 'MAR' or mask_type == 'MNAR':
        X = simulate_nan(df_X.to_numpy(), 0.3, mask_type)
        train_mask = ~X['mask'].astype(bool)
        train_edge_mask = torch.from_numpy(train_mask.reshape(-1))
    elif mask_type == 'MCAR':
        train_edge_mask = get_known_mask(train_edge_prob, int(edge_attr.shape[0]/2))

    double_train_edge_mask = torch.cat((train_edge_mask, train_edge_mask), dim=0)
    #mask edges based on the generated train_edge_mask
    #train_edge_index is known, test_edge_index in unknwon, i.e. missing
    train_edge_index, train_edge_attr = mask_edge(edge_index, edge_attr,
                                                double_train_edge_mask, True)
    if initial == 'rule':
        mask = train_edge_mask.reshape(x.shape[0] - x.shape[1],x.shape[1])
        df_X_fp = features.copy(deep=True).values

        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if(mask[i][j].item()):
                    continue
                else:
                    features[j][i] = None
        features = cluster_features(features,dataset)
        features = add_tag(features).to_numpy()

        fp_list = []
        for i in range(mask.shape[0]):
            fp_list.append(features[i][mask[i]].tolist())
        # association rules mining
        assRuleDf = fpgrowth(fp_list, confidence)
        # lift>1
        rules_other, rules_other_confidence, rules_other_lift = get_all_rule(assRuleDf)

        add_edge_start, add_edge_end = get_rules_edge_with_mask(rules_other,features)

    elif initial == 'random':
        random_edges = np.floor(1.0 * len(features))
        add_edge_start, add_edge_end = get_edge_with_random(features,confidence,random_edges)
        # from sklearn.metrics.pairwise import cosine_similarity
        # m1 = np.mat(df_X)
        # cos = cosine_similarity(m1)
        # result = []
        # for i in range(len(add_edge_start)):
        #     result.append(cos[add_edge_start[i]][add_edge_end[i]])
    elif initial == 'cos':
        index = features.shape[0]
        cos_list = [[0 for i in range(index)] for j in range(index)]
        mask = train_edge_mask.reshape(x.shape[0] - x.shape[1],x.shape[1])
        df_X_fp = features.copy(deep=True).values
        for index0 in range(index):
            for index1 in range(index0+1, index):
                cos_list[index0][index1] = get_cos(index0, index1, mask, df_X_fp)
        topk_index = list(np.argpartition(list(np.array(cos_list).flatten()), -index))[-index:]
        adj_sampled = []
        m1 = np.mat(df_X)
        cos = cosine_similarity(m1)
        result = []
        for i in topk_index:
            row = i // index
            col = i % index
            result.append(cos[row][col])
            adj_sampled.append([row.item(),col.item()])
        adj_sampled = torch.tensor(adj_sampled).T
    elif initial == 'cos_complete':
        m1 = np.mat(df_X)
        cos = cosine_similarity(m1)
        cos = np.triu(cos,1)
        row_num = df_X.shape[0]
        all_index = 5000
        alpha = 1.0
        positive = int(alpha*all_index)
        topk_index = list(np.argpartition(list(np.array(cos).flatten()), -positive))[-positive:]
        adj_sampled = []
        for i in topk_index:
            row = i // row_num
            col = i % row_num
            adj_sampled.append([row.item(),col.item()])
        # negative samples
        negative = int((1-alpha)*all_index)
        sample_list = list(np.arange(row_num))
        for i in range(negative):
            edge = random.sample(sample_list,2)
            while(edge in adj_sampled):
                edge = random.sample(sample_list,2)
            adj_sampled.append([edge[0],edge[1]])
        adj_sampled = torch.tensor(adj_sampled).T

        
    if initial == 'random' or initial == 'fixed':
        edge_start = (train_edge_index[0][0:int(train_edge_index[0].shape[0]/2)]).numpy().tolist() + add_edge_start
        edge_end = (train_edge_index[1][0:int(train_edge_index[1].shape[0]/2)]).numpy().tolist() + add_edge_end
        edge_start_ = edge_start + edge_end
        edge_end_ = edge_end + edge_start
        train_edge_index_ = torch.tensor([edge_start_, edge_end_], dtype=int)
        obob_edge_start = add_edge_start+add_edge_end
        obob_edge_end = add_edge_end+add_edge_start
        obob_edge_index = torch.tensor([obob_edge_start,obob_edge_end])
    elif initial == 'cos' or initial == 'cos_complete':
        obob_edge_index = adj_sampled

    obob_adj_train = get_obob_adj_matrix(df_X, obob_edge_index)
    obob_adj_orig = scipysp_to_pytorchsp(obob_adj_train).to_dense()  # 边的矩阵的torch.tensor表示
    obob_adj_norm = normalize_adj_(obob_adj_train)

    train_labels = train_edge_attr[:int(train_edge_attr.shape[0]/2),0]
    test_edge_index, test_edge_attr = mask_edge(edge_index, edge_attr,
                                                ~double_train_edge_mask, True)
    test_labels = test_edge_attr[:int(test_edge_attr.shape[0]/2),0]
    #mask the y-values during training, i.e. how we split the training and test sets
    train_y_mask = get_known_mask(train_y_prob, y.shape[0])
    test_y_mask = ~train_y_mask

    data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr,
            train_y_mask=train_y_mask, test_y_mask=test_y_mask,
            train_edge_index=train_edge_index,train_edge_attr=train_edge_attr,
            train_edge_mask=train_edge_mask,train_labels=train_labels,
            test_edge_index=test_edge_index,test_edge_attr=test_edge_attr,
            test_edge_mask=~train_edge_mask,test_labels=test_labels, 
            df_X=df_X,df_y=df_y,
            edge_attr_dim=train_edge_attr.shape[-1],
            user_num=df_X.shape[0],
            obob_edge_index = obob_edge_index,
            # obob_adj_train = obob_adj_train,
            obob_adj_norm = obob_adj_norm,
            obob_adj_orig=obob_adj_orig,
            mask=train_edge_mask
            )

    if split_sample_ratio > 0.:
        if split_by == 'y':
            sorted_y, sorted_y_index = torch.sort(torch.reshape(y,(-1,)))
        elif split_by == 'random':
            sorted_y_index = torch.randperm(y.shape[0])
        lower_y_index = sorted_y_index[:int(np.floor(y.shape[0]*split_sample_ratio))]
        higher_y_index = sorted_y_index[int(np.floor(y.shape[0]*split_sample_ratio)):]
        # here we don't split x, only split edge
        # train
        half_train_edge_index = train_edge_index[:,:int(train_edge_index.shape[1]/2)];
        lower_train_edge_mask = []
        for node_index in half_train_edge_index[0]:
            if node_index in lower_y_index:
                lower_train_edge_mask.append(True)
            else:
                lower_train_edge_mask.append(False)
        lower_train_edge_mask = torch.tensor(lower_train_edge_mask)
        double_lower_train_edge_mask = torch.cat((lower_train_edge_mask, lower_train_edge_mask), dim=0)
        lower_train_edge_index, lower_train_edge_attr = mask_edge(train_edge_index, train_edge_attr,
                                                double_lower_train_edge_mask, True)
        lower_train_labels = lower_train_edge_attr[:int(lower_train_edge_attr.shape[0]/2),0]
        higher_train_edge_index, higher_train_edge_attr = mask_edge(train_edge_index, train_edge_attr,
                                                ~double_lower_train_edge_mask, True)
        higher_train_labels = higher_train_edge_attr[:int(higher_train_edge_attr.shape[0]/2),0]
        # test
        half_test_edge_index = test_edge_index[:,:int(test_edge_index.shape[1]/2)];
        lower_test_edge_mask = []
        for node_index in half_test_edge_index[0]:
            if node_index in lower_y_index:
                lower_test_edge_mask.append(True)
            else:
                lower_test_edge_mask.append(False)
        lower_test_edge_mask = torch.tensor(lower_test_edge_mask)
        double_lower_test_edge_mask = torch.cat((lower_test_edge_mask, lower_test_edge_mask), dim=0)
        lower_test_edge_index, lower_test_edge_attr = mask_edge(test_edge_index, test_edge_attr,
                                                double_lower_test_edge_mask, True)
        lower_test_labels = lower_test_edge_attr[:int(lower_test_edge_attr.shape[0]/2),0]
        higher_test_edge_index, higher_test_edge_attr = mask_edge(test_edge_index, test_edge_attr,
                                                ~double_lower_test_edge_mask, True)
        higher_test_labels = higher_test_edge_attr[:int(higher_test_edge_attr.shape[0]/2),0]


        data.lower_y_index = lower_y_index
        data.higher_y_index = higher_y_index
        data.lower_train_edge_index = lower_train_edge_index
        data.lower_train_edge_attr = lower_train_edge_attr
        data.lower_train_labels = lower_train_labels
        data.higher_train_edge_index = higher_train_edge_index
        data.higher_train_edge_attr = higher_train_edge_attr
        data.higher_train_labels = higher_train_labels
        data.lower_test_edge_index = lower_test_edge_index
        data.lower_test_edge_attr = lower_test_edge_attr
        data.lower_test_labels = lower_train_labels
        data.higher_test_edge_index = higher_test_edge_index
        data.higher_test_edge_attr = higher_test_edge_attr
        data.higher_test_labels = higher_test_labels
        
    return data

def load_data(args):
    uci_path = osp.dirname(osp.abspath(inspect.getfile(inspect.currentframe())))
    df_np = np.loadtxt(uci_path+'/raw_data/{}/data/data.txt'.format(args.data))
    df_y = pd.DataFrame(df_np[:, -1:])
    df_X = pd.DataFrame(df_np[:, :-1])

    if args.data in ['concrete','protein','power','wine','heart','DOW30','diabetes']:
        confidence = 0.6
    elif args.data in ['ecommerce']:
        confidence = 0.5
    elif args.data in ['housing']:
        confidence = 0.7
    else:
        confidence = 0

    if not hasattr(args,'split_sample'):
        args.split_sample = 0
    data = get_data(df_X, df_y, args.node_mode, args.train_edge, args.split_sample, args.split_by, args.train_y, confidence, args.seed, args.data)
    return data


def get_data_fix_mask(df_X, train_mask, confidence=0, node_mode=0, train_edge_prob=0.7, seed=0, dataset=None, normalize=True, df_y=None):
    if df_y is None:
        #dummy df_y for compatibility
        df_y = pd.DataFrame(np.zeros(df_X.shape[0]))
    
    if len(df_y.shape)==1:
        df_y = df_y.to_numpy()
    elif len(df_y.shape)==2:
        df_y = df_y[0].to_numpy()
    features = pd.DataFrame(df_X.values)

    # if normalize:
    #     x = df_X.values
    #     min_max_scaler = preprocessing.MinMaxScaler()
    #     x_scaled = min_max_scaler.fit_transform(x)
    #     df_X = pd.DataFrame(x_scaled)

    # created fully connected grpah
    edge_start, edge_end = create_edge(df_X)
    edge_index = torch.tensor([edge_start, edge_end], dtype=int)
    edge_attr = torch.tensor(create_edge_attr(df_X), dtype=torch.float)
    node_init = create_node(df_X, node_mode, df_y) 
    x = torch.tensor(node_init, dtype=torch.float)
    y = torch.tensor(df_y, dtype=torch.float)
    
    #set seed to fix known/unknwon edges
    torch.manual_seed(seed)
    # # select missing mechanism
    # if mask_type == 'MAR' or mask_type == 'MNAR':
    #     X = simulate_nan(df_X.to_numpy(), 0.3, mask_type)
    #     train_mask = ~X['mask'].astype(bool)
    #     train_edge_mask = torch.from_numpy(train_mask.reshape(-1))
    # elif mask_type == 'MCAR':
    #     train_edge_mask = get_known_mask(train_edge_prob, int(edge_attr.shape[0]/2))
    train_edge_mask = torch.tensor(train_mask).reshape(-1)

    '''
    Important: need to flip the mask 
    For our mask, True means unknown, False means known 
    For here, True means known, False means unknown
    '''
    train_edge_mask = ~train_edge_mask #flip the mask
    double_train_edge_mask = torch.cat((train_edge_mask, train_edge_mask), dim=0)

    #mask edges based on the generated train_edge_mask
    #train_edge_index is known, test_edge_index in unknwon, i.e. missing
    train_edge_index, train_edge_attr = mask_edge(edge_index, edge_attr,
                                                double_train_edge_mask, True)
    if initial == 'rule':
        mask = train_edge_mask.reshape(x.shape[0] - x.shape[1],x.shape[1])
        df_X_fp = features.copy(deep=True).values

        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if(mask[i][j].item()):
                    continue
                else:
                    features[j][i] = None
        features = cluster_features(features,dataset)
        features = add_tag(features).to_numpy()

        fp_list = []
        for i in range(mask.shape[0]):
            fp_list.append(features[i][mask[i]].tolist())
        # association rules mining
        assRuleDf = fpgrowth(fp_list, confidence)
        # lift>1
        rules_other, rules_other_confidence, rules_other_lift = get_all_rule(assRuleDf)

        add_edge_start, add_edge_end = get_rules_edge_with_mask(rules_other,features)

    elif initial == 'random':
        random_edges = np.floor(1.0 * len(features))
        add_edge_start, add_edge_end = get_edge_with_random(features,confidence,random_edges)
        # from sklearn.metrics.pairwise import cosine_similarity
        # m1 = np.mat(df_X)
        # cos = cosine_similarity(m1)
        # result = []
        # for i in range(len(add_edge_start)):
        #     result.append(cos[add_edge_start[i]][add_edge_end[i]])
    elif initial == 'cos':
        index = features.shape[0]
        cos_list = [[0 for i in range(index)] for j in range(index)]
        mask = train_edge_mask.reshape(x.shape[0] - x.shape[1],x.shape[1])
        df_X_fp = features.copy(deep=True).values
        for index0 in range(index):
            for index1 in range(index0+1, index):
                cos_list[index0][index1] = get_cos(index0, index1, mask, df_X_fp)
        topk_index = list(np.argpartition(list(np.array(cos_list).flatten()), -index))[-index:]
        adj_sampled = []
        m1 = np.mat(df_X)
        cos = cosine_similarity(m1)
        result = []
        for i in topk_index:
            row = i // index
            col = i % index
            result.append(cos[row][col])
            adj_sampled.append([row.item(),col.item()])
        adj_sampled = torch.tensor(adj_sampled).T
    elif initial == 'cos_complete':
        m1 = np.mat(df_X)
        cos = cosine_similarity(m1)
        cos = np.triu(cos,1)
        row_num = df_X.shape[0]
        all_index = 5000
        alpha = 1.0
        positive = int(alpha*all_index)
        topk_index = list(np.argpartition(list(np.array(cos).flatten()), -positive))[-positive:]
        adj_sampled = []
        for i in topk_index:
            row = i // row_num
            col = i % row_num
            adj_sampled.append([row.item(),col.item()])
        # negative samples
        negative = int((1-alpha)*all_index)
        sample_list = list(np.arange(row_num))
        for i in range(negative):
            edge = random.sample(sample_list,2)
            while(edge in adj_sampled):
                edge = random.sample(sample_list,2)
            adj_sampled.append([edge[0],edge[1]])
        adj_sampled = torch.tensor(adj_sampled).T

        
    if initial == 'random' or initial == 'fixed':
        edge_start = (train_edge_index[0][0:int(train_edge_index[0].shape[0]/2)]).numpy().tolist() + add_edge_start
        edge_end = (train_edge_index[1][0:int(train_edge_index[1].shape[0]/2)]).numpy().tolist() + add_edge_end
        edge_start_ = edge_start + edge_end
        edge_end_ = edge_end + edge_start
        train_edge_index_ = torch.tensor([edge_start_, edge_end_], dtype=int)
        obob_edge_start = add_edge_start+add_edge_end
        obob_edge_end = add_edge_end+add_edge_start
        obob_edge_index = torch.tensor([obob_edge_start,obob_edge_end])
    elif initial == 'cos' or initial == 'cos_complete':
        obob_edge_index = adj_sampled

    obob_adj_train = get_obob_adj_matrix(df_X, obob_edge_index)
    obob_adj_orig = scipysp_to_pytorchsp(obob_adj_train).to_dense()  # 边的矩阵的torch.tensor表示
    obob_adj_norm = normalize_adj_(obob_adj_train)

    train_labels = train_edge_attr[:int(train_edge_attr.shape[0]/2),0]
    test_edge_index, test_edge_attr = mask_edge(edge_index, edge_attr,
                                                ~double_train_edge_mask, True)
    test_labels = test_edge_attr[:int(test_edge_attr.shape[0]/2),0]
    #mask the y-values during training, i.e. how we split the training and test sets
    #train_y_mask = get_known_mask(train_y_prob, y.shape[0])
    #test_y_mask = ~train_y_mask

    data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr,
            #train_y_mask=train_y_mask, test_y_mask=test_y_mask,
            train_edge_index=train_edge_index,train_edge_attr=train_edge_attr,
            train_edge_mask=train_edge_mask,train_labels=train_labels,
            test_edge_index=test_edge_index,test_edge_attr=test_edge_attr,
            test_edge_mask=~train_edge_mask,test_labels=test_labels, 
            df_X=df_X,df_y=df_y,
            edge_attr_dim=train_edge_attr.shape[-1],
            user_num=df_X.shape[0],
            obob_edge_index = obob_edge_index,
            # obob_adj_train = obob_adj_train,
            obob_adj_norm = obob_adj_norm,
            obob_adj_orig=obob_adj_orig,
            mask=train_edge_mask
            )

    
    return data
