import os
import sys
import csv
import wget
import zipfile
import numpy as np
import pandas as pd
import torch
from torch import nn
import torchvision
from torchvision import transforms, datasets
from torch import distributions
import ssl
from tqdm import tqdm
import pdb
'''
Helper functions for MCFlow
'''

def endtoend_train(flow, nn_model, nf_optimizer, nn_optimizer, loader, args, epoch):

    nf_totalloss = 0
    totalloss = 0
    total_log_loss = 0
    total_imputing = 0
    loss_func = nn.MSELoss(reduction='none')

    pbar = tqdm(loader, total=len(loader))
    pbar.set_description(f"Epoch {epoch}/{args.n_epochs}")

    for index, (vectors, labels) in enumerate(pbar):
        if args.use_cuda:
            vectors = vectors.cuda()
            labels[0] = labels[0].cuda()
            labels[1] = labels[1].cuda()

        z, nf_loss = flow.log_prob(vectors, args)
        nf_totalloss += nf_loss.item()
        z_hat = nn_model(z)
        x_hat = flow.inverse(z_hat)
        _, log_p = flow.log_prob(x_hat, args)

        batch_loss = torch.sum(loss_func(x_hat, labels[0]) * (1 - labels[1]))
        #batch_loss_2 = torch.sum((x_hat - labels[0]) ** 2 * (1-labels[1])) #equivalent to the above
        total_imputing += np.sum(1-labels[1].cpu().numpy())

        log_lss = log_p
        total_log_loss += log_p.item()
        totalloss += batch_loss.item()
        batch_loss += log_lss
        nf_loss.backward(retain_graph=True)
        nf_optimizer.step()
        nf_optimizer.zero_grad()
        batch_loss.backward()
        nn_optimizer.step()
        nn_optimizer.zero_grad()

    index+=1
    return totalloss, total_log_loss/index, nf_totalloss/index

def get_filled_data(flow, nn_model, data_loader, args):
    filled_data = []
    
    totalloss = 1
    total_imputing = 0
    for _, (vectors, labels) in enumerate(data_loader):
        if args.use_cuda:
            vectors = vectors.cuda()
            labels[0] = labels[0].cuda()
            labels[1] = labels[1].cuda()

        z, nf_loss = flow.log_prob(vectors, args)
        z_hat = nn_model(z)
        x_hat = flow.inverse(z_hat)
        filled_data.append(torch.clamp(x_hat, min=0, max=1).detach().cpu().numpy())
        #filled_data.append(x_hat.detach().cpu().numpy())

        x_hat_trun = torch.clamp(x_hat, min=0, max=1)
        batch_loss = torch.sum((x_hat_trun - labels[0]) ** 2 * labels[1])
        total_imputing += np.sum(labels[1].cpu().numpy())
        totalloss+=batch_loss.item()
    
    # Concatenate all the batches
    filled_data = np.concatenate(filled_data, axis=0)
    return filled_data, totalloss/total_imputing #for debug


def endtoend_test(flow, nn_model, data_loader, args):
    totalloss = 1
    nf_totalloss = 0
    total_imputing = 0
    loss = nn.MSELoss(reduction='none')
    
    for index, (vectors, labels) in enumerate(data_loader):
        
        if args.use_cuda:
            vectors = vectors.cuda()
            labels[0] = labels[0].cuda()
            labels[1] = labels[1].cuda()

        z, nf_loss = flow.log_prob(vectors, args)
        nf_totalloss += nf_loss.item()

        z_hat = nn_model(z)

        x_hat = flow.inverse(z_hat)

        #batch_loss_ori = torch.sum(loss(torch.clamp(x_hat, min=0, max=1), labels[0]) * labels[1])
        x_hat_trun = torch.clamp(x_hat, min=0, max=1)
        batch_loss = torch.sum((x_hat_trun - labels[0]) ** 2 * labels[1])
        total_imputing += np.sum(labels[1].cpu().numpy())
        totalloss+=batch_loss.item()
    
    index+=1
    return totalloss/total_imputing, nf_totalloss/index


def create_mask(shape):
    zeros = int(shape/2)
    ones = shape - zeros
    lst = []
    for i in range(shape):
        if zeros > 0 and ones > 0:
            if np.random.uniform() > .5:
                lst.append(0)
                zeros -= 1
            else:
                lst.append(1)
                ones -= 1
        elif zeros > 0:
            lst.append(0)
            zeros -= 1
        else:
            lst.append(1)
            ones -= 1
    return np.asarray(lst)

def init_flow_model(num_neurons, num_layers, init_flow, data_shape, args):

    nets = lambda: nn.Sequential(nn.Linear(data_shape, num_neurons), nn.LeakyReLU(), nn.Linear(num_neurons, num_neurons), nn.LeakyReLU(), nn.Linear(num_neurons, num_neurons),
        nn.LeakyReLU(), nn.Linear(num_neurons, data_shape), nn.Tanh())

    nett = lambda: nn.Sequential(nn.Linear(data_shape, num_neurons), nn.LeakyReLU(), nn.Linear(num_neurons, num_neurons), nn.LeakyReLU(),
        nn.Linear(num_neurons, num_neurons),  nn.LeakyReLU(), nn.Linear(num_neurons, data_shape))

    mask = []
    for idx in range(num_layers):
        msk = create_mask(data_shape)
        mask.append(msk)
        mask.append(1-msk)


    masks = torch.from_numpy(np.asarray(mask)).float()
    if args.use_cuda:
        masks = masks.cuda()
    prior = distributions.MultivariateNormal(torch.zeros(data_shape), torch.eye(data_shape))
    flow = init_flow(nets, nett, masks, prior)
    if args.use_cuda:
        flow.cuda()

    return flow


def inference_imputation_networks(nn, nf, data, args):
    lst = []

    batch_sz = 256
    iterations = int(data.shape[0]/batch_sz)
    left_over = data.shape[0] - batch_sz * iterations

    with torch.no_grad():
        for idx in range(iterations):
            rows = data[int(idx*batch_sz):int((idx+1)*batch_sz)]
            if args.use_cuda:
                rows = torch.from_numpy(rows).float().cuda()
            else:
                rows = torch.from_numpy(rows).float()

            z = nf(rows)[0]
            z_hat = nn(z)
            x_hat = nf.inverse(z_hat)

            lst.append(np.clip(x_hat.cpu().numpy(),0,1))


        rows = data[int((idx+1)*batch_sz):]
        if args.use_cuda:
            rows = torch.from_numpy(rows).float().cuda()
        else:
            rows = torch.from_numpy(rows).float()

        z = nf(rows)[0]
        z_hat = nn(z)
        x_hat = nf.inverse(z_hat)

        lst.append(np.clip(x_hat.cpu().numpy(), 0, 1))

    final_lst = []
    for idx in range(len(lst)):
        for element in lst[idx]:
            final_lst.append(element)

    return final_lst

def inference_img_imputation_networks(nn, nf, data, mask, original_dat, args):

    batch_sz = 256
    iterations = int(len(data)/batch_sz)
    left_over = len(data) - batch_sz * iterations
    ones = np.ones((256, data[0].shape[0]))

    with torch.no_grad():
        for idx in range(iterations):
            begin =int(idx*batch_sz)
            end =int((idx+1)*batch_sz)
            rows = np.asarray(data[begin:end])
            if args.use_cuda:
                rows = torch.from_numpy(rows).float().cuda()
            else:
                rows = torch.from_numpy(rows).float()

            z = nf(rows)[0]
            z_hat = nn(z)
            x_hat = nf.inverse(z_hat)
            x_hat = np.clip(x_hat.cpu().numpy(),0,1)
            data[begin:end] = (ones-mask[begin:end]) * original_dat[begin:end] +  mask[begin:end] * x_hat

        rows = np.asarray(data[-left_over:])
        if args.use_cuda:
            rows = torch.from_numpy(rows).float().cuda()
        else:
            rows = torch.from_numpy(rows).float()
        ones = np.ones((left_over, data[0].shape[0]))

        z = nf(rows)[0]
        z_hat = nn(z)
        x_hat = nf.inverse(z_hat)
        x_hat = np.clip(x_hat.cpu().numpy(),0,1)
        data[-left_over:] = (ones-mask[-left_over:]) * original_dat[-left_over:] +  mask[-left_over:] * x_hat



def create_k_fold_mask(seed, mask):
    #I will create different folds based on the seed
    fold_sz = int(mask.shape[0]*.2) #5 folds

    if seed == 0:
        mask_tr = mask[fold_sz:]
        mask_te = mask[:fold_sz]
    elif seed == 1:
        mask_tr = np.concatenate((mask[:fold_sz], mask[int(fold_sz*2):]))
        mask_te = mask[fold_sz:int(fold_sz*2)]
    elif seed == 2:
        mask_tr = np.concatenate((mask[:int(fold_sz*2)], mask[int(fold_sz*3):]))
        mask_te = mask[int(fold_sz*2):int(fold_sz*3)]
    elif seed == 3:
        mask_tr = np.concatenate((mask[:int(fold_sz*3)], mask[int(fold_sz*4):]))
        mask_te = mask[int(fold_sz*3):int(fold_sz*4)]
    elif seed == 4:
        mask_tr = mask[:int(fold_sz*4)]
        mask_te = mask[int(4*fold_sz):]
    else:
        print("incorrect seed for the fold")
        sys.exit()

    return mask_tr, mask_te

def create_img_dropout_masks(drp_percent, path, img_shp, num_tr, num_te):
    train_mask = []
    test_mask = []

    num_channels = img_shp[0]

    for idx in range(num_tr):
        sample = []
        for r_idx in range(img_shp[1]):
            for c_idx in range(img_shp[2]):
                if np.random.uniform() < drp_percent:
                    sample.append(1)
                else:
                    sample.append(0)
        if num_channels == 1:
            train_mask.append(np.asarray(sample))
        else:
            train_mask.append(np.asarray(sample + sample + sample))

    for idx in range(num_te):
        sample = []
        for r_idx in range(img_shp[1]):
            for c_idx in range(img_shp[2]):
                if np.random.uniform() < drp_percent:
                    sample.append(1)
                else:
                    sample.append(0)
        if num_channels == 1:
            test_mask.append(np.asarray(sample))
        else:
            test_mask.append(np.asarray(sample + sample + sample))

    return train_mask, test_mask


def initialize_nneighbor_radnommat(dta, msk, shape):

    data = dta.copy()
    for idx, el in enumerate(data):
        #reshape the data
        updates_required_lst = []
        element = np.reshape(el, shape)
        updater = element.copy()
        mask = np.reshape(msk[idx], shape)[0]

        for r_idx in range(mask.shape[0]):
            for c_idx in range(mask.shape[1]):
                if mask[r_idx][c_idx] == 1:
                    updates_required_lst.append((r_idx, c_idx))

        for point in updates_required_lst:
            layer = 1
            neighbors = []
            while len(neighbors) == 0:
                corners = [(point[0] + layer, point[1] + layer),
                    (point[0] - layer, point[1] + layer),
                    (point[0] + layer, point[1] - layer),
                    (point[0] - layer, point[1] - layer)]

                #row check -- need to check if the mask is zero there or that the data exists
                for _row in range(corners[1][0], corners[0][0]):
                    if _row >= 0 and _row < mask.shape[0] and corners[0][1] >=0 and corners[0][1] < mask.shape[0]:
                        if mask[_row][corners[0][1]] == 0:
                            neighbors.append((_row, corners[0][1]))

                #column check
                for _column in range(corners[2][1], corners[0][1]):
                    if _column >= 0 and _column < mask.shape[1] and corners[0][0] >=0 and corners[0][0] < mask.shape[0]:
                        if mask[corners[0][0]][_column] == 0:
                            neighbors.append((corners[0][0], _column))

                #row check
                for _row in range(corners[3][0], corners[2][0]):
                    if _row >= 0 and _row < mask.shape[0] and corners[2][1] >=0 and corners[2][1] < mask.shape[0]:
                        if mask[_row][corners[2][1]] == 0:
                            neighbors.append((_row, corners[2][1]))

                #column check
                for _column in range(corners[3][1], corners[1][1]):
                    if _column >= 0 and _column < mask.shape[1] and corners[1][0] >=0 and corners[1][0] < mask.shape[0]:
                        if mask[corners[1][0]][_column] == 0:
                            neighbors.append((corners[1][0], _column))

                layer+=1

            loc = np.random.randint(len(neighbors))
            #Fill the values based on the channels
            if shape[0] == 1:
                updater[0][point[0]][point[1]] = element[0][neighbors[loc][0]][neighbors[loc][1]]
            else:
                updater[0][point[0]][point[1]] = element[0][neighbors[loc][0]][neighbors[loc][1]]
                updater[1][point[0]][point[1]] = element[1][neighbors[loc][0]][neighbors[loc][1]]
                updater[2][point[0]][point[1]] = element[2][neighbors[loc][0]][neighbors[loc][1]]

        data[idx] = updater.flatten()

    return data


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def fill_img_missingness(tr_data, te_data, mask_tr, mask_te, shape, type):
    if type == 0:
        train = initialize_nneighbor_radnommat(tr_data, mask_tr, shape)
        test = initialize_nneighbor_radnommat(te_data, mask_te, shape)
    else:
        #pixelwise imputation
        print("not yet implemented")
        sys.exit()
        train = initialize_pixelwise_radnommat(tr_data, mask_tr, shape) #still needs to be implemented
        test = initialize_pixelwise_radnommat(te_data, mask_te, shape) #still needs to be implemented

    return train, test



def create_k_fold(matrix, seed):
    #I will create different folds based on the seed
    fold_sz = int(matrix.shape[0]*.2) #5 folds

    if seed == 0:
        train = matrix[fold_sz:]
        test = matrix[:fold_sz]
    elif seed == 1:
        train = np.concatenate((matrix[:fold_sz], matrix[int(fold_sz*2):]))
        test = matrix[fold_sz:int(fold_sz*2)]
    elif seed == 2:
        train = np.concatenate((matrix[:int(fold_sz*2)], matrix[int(fold_sz*3):]))
        test = matrix[int(fold_sz*2):int(fold_sz*3)]
    elif seed == 3:
        train = np.concatenate((matrix[:int(fold_sz*3)], matrix[int(fold_sz*4):]))
        test = matrix[int(fold_sz*3):int(fold_sz*4)]
    elif seed == 4:
        train = matrix[:int(fold_sz*4)]
        test = matrix[int(4*fold_sz):]
    else:
        print("incorrect seed for the fold")
        sys.exit()

    return train, test


def path_to_matrix(path):
    if path == 'news':
        try:
            df = pd.read_csv('./data/OnlineNewsPopularity/OnlineNewsPopularity.csv')
            matrix = df.values[:,1:]
        except:
            print("\nDownloading OnlineNewsPopularity dataset\n")
            ssl._create_default_https_context = ssl._create_unverified_context
            wget.download('https://archive.ics.uci.edu/ml/machine-learning-databases/00332/OnlineNewsPopularity.zip')
            with zipfile.ZipFile('OnlineNewsPopularity.zip', 'r') as zip_ref:
                zip_ref.extractall('./data/')
            os.remove("OnlineNewsPopularity.zip")
            if os.path.exists('./data/OnlineNewsPopularity/OnlineNewsPopularity.csv'):
                print("\nSuccessfully downloaded OnlineNewsPopularity dataset from the UCI database")
                df = pd.read_csv('./data/OnlineNewsPopularity/OnlineNewsPopularity.csv')
                matrix = df.values[:,1:]
            else:
                print("\n\nError downloading UCI database please extract OnlineNewsPopularity.zip in the data folder")
                print("Donwload OnlineNewsPopularity.zip at https://archive.ics.uci.edu/ml/machine-learning-databases/00332/OnlineNewsPopularity.zip")
                sys.exit()
        return matrix
    elif path == 'mnist':
        mnist_trainset = datasets.MNIST(root='./data', train=True, download=True)
        mnist_testset = datasets.MNIST(root='./data', train=False, download=True)
        mnist_train = []
        mnist_test = []
        for idx in range(len(mnist_trainset)):
            mnist_train.append(np.array(mnist_trainset[idx][0]).flatten() / 255)

        for idx in range(len(mnist_testset)):
            mnist_test.append(np.array(mnist_testset[idx][0]).flatten() /255)
        return mnist_train, mnist_test, (1, 28, 28)
    else:
        print("Not a valid dataset\n\n")
        print("Valid datasets include: \nnews")
        print("mnist")
        sys.exit()


def preprocess(data):
    maxs = np.zeros(data.shape[1])
    mins = np.zeros(data.shape[1])
    for idx, row in enumerate(data.T):
        maxs[idx] = row.max()
        mins[idx] = row.min()
    dat = []
    for idx, row in enumerate(data):
        rw = []
        for idx_v, value in enumerate(row):
            rw.append((value - mins[idx_v]) / (maxs[idx_v] - mins[idx_v]))
        dat.append(np.asarray(rw))

    return np.asarray(dat), maxs, mins



def make_random_matrix(matrix, unique_values, path):
    random = np.zeros(matrix.shape)
    for r_idx in range(random.shape[0]):
        for c_idx in range(random.shape[1]):
            loc = np.random.randint(int(unique_values[c_idx].shape[0])) #Get a value from the other data available
            random[r_idx][c_idx] += unique_values[c_idx][loc] #replace this with the values that exist in each column

    return random



def fill_missingness(matrix, mask, unique_values, path, seed=0):
    random_mat = make_random_matrix(matrix, unique_values, path)
    matrix = np.nan_to_num((1-mask) * matrix) + mask * random_mat
    return create_k_fold(matrix, seed)

def fill_missingness_train_test(matrix, mask, unique_values, path):
    random_mat = make_random_matrix(matrix, unique_values, path)
    matrix = np.nan_to_num((1-mask) * matrix) + mask * random_mat
    return matrix

def make_static_mask(drp_percent, seed, path, matrix):
    mask = np.zeros(matrix.shape)
    if os.path.exists('./masks/' + path +'mask.npy'):
        mask = np.load('./masks/' + path +'mask.npy')
    else:
        for r_idx, row in enumerate(mask):
            for c_idx, element in enumerate(row):
                if np.random.uniform() < drp_percent:
                    mask[r_idx][c_idx] += 1

        np.save('./masks/' + path +'mask.npy', mask)

    return mask
