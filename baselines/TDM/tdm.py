import numpy as np
import torch


from .utils import nanmean, MAE, RMSE

import logging

import ot

import pdb

class TDM():
    
    def __init__(self,
                 projector,
                 im_lr=1e-2,
                 proj_lr=1e-2,
                 opt=torch.optim.RMSprop, 
                 niter=2000,
                 batchsize=128,
                 n_pairs=1,
                 noise=0.1,
                 save_dir_training=None):

        self.im_lr = im_lr
        self.proj_lr = proj_lr
        self.opt = opt
        self.niter = niter
        self.batchsize = batchsize
        self.n_pairs = n_pairs
        self.noise = noise
        self.projector = projector

        self.save_dir_training = save_dir_training


    def fit_transform(self, X, verbose=True, report_interval=500, X_true=None):
       
        X = X.clone()
        n, d = X.shape

        
        if self.batchsize > n // 2:
            e = int(np.log2(n // 2))
            self.batchsize = 2**e
            if verbose:
                logging.info(f"Batchsize larger that half size = {len(X) // 2}. Setting batchsize to {self.batchsize}.")

        mask = torch.isnan(X).double()

        torch.autograd.set_detect_anomaly(True)


        if torch.sum(mask) < 1.0:
            is_no_missing = True
        else:
            is_no_missing = False

        X_filled = X.detach().clone()   

        if not is_no_missing:
            imps = (self.noise * torch.randn(mask.shape).double() + nanmean(X, 0))[mask.bool()]
            imps.requires_grad = True
            im_optimizer = self.opt([imps], lr=self.im_lr)
            X_filled[mask.bool()] = imps

        proj_optimizer = self.opt([p for p in self.projector.parameters()], lr=self.proj_lr)

        if X_true is not None:
            maes = np.zeros(self.niter)
            rmses = np.zeros(self.niter)

        for i in range(self.niter):

            X_filled = X.detach().clone()

            if not is_no_missing:
                X_filled[mask.bool()] = imps


            proj_loss = 0
            im_loss = 0


            for _ in range(self.n_pairs):

                idx1 = np.random.choice(n, self.batchsize, replace=False)
                idx2 = np.random.choice(n, self.batchsize, replace=False)

                X1 = X_filled[idx1]
                X2 = X_filled[idx2]


                X1_p, _ = self.projector(X1)
                X2_p, _ = self.projector(X2)


                M_p = torch.cdist(X1_p, X2_p, p=2)

                a1_p = torch.ones(X1.shape[0]) / X1.shape[0]
                a2_p = torch.ones(X2.shape[0]) / X2.shape[0]
                a1_p.requires_grad = False
                a2_p.requires_grad = False
                ot_p = ot.emd2(a1_p, a2_p, M_p)


                im_loss = im_loss + ot_p
                proj_loss = proj_loss + ot_p 


            if torch.isnan(im_loss).any() or torch.isinf(im_loss).any():
                logging.info("im_loss Nan or inf loss")
                break

            if torch.isnan(proj_loss).any() or torch.isinf(proj_loss).any():
                logging.info("proj_loss Nan or inf loss")
                break

            if not is_no_missing:
                im_optimizer.zero_grad()
                im_loss.backward(retain_graph=True)
                im_optimizer.step()
            


            proj_optimizer.zero_grad()
            proj_loss.backward()
            proj_optimizer.step()


            maes[i] = MAE(X_filled, X_true, mask).item() 
            rmses[i] = RMSE(X_filled, X_true, mask).item()
            if verbose and (i % report_interval == 0):

                if X_true is not None:
                    #maes[i] = MAE(X_filled, X_true, mask).item() 
                    #rmses[i] = RMSE(X_filled, X_true, mask).item()

                    logging.info(f'Iteration {i}:\t Imputer Loss: {im_loss.item():.4f}\t '
                                 f'Projector Loss: {proj_loss.item():.4f}\t '
                                 f'Validation MAE: {maes[i]:.4f}\t'
                                 f'RMSE: {rmses[i]:.4f}')


                else:
                    logging.info(f'Iteration {i}:\t Imputer Loss: {im_loss.item():.4f}\t '
                                 f'Projector Loss: {proj_loss.item():.4f}\t ')
              
            

        X_filled = X.detach().clone()
        if not is_no_missing:
            X_filled[mask.bool()] = imps

        if X_true is not None:
            return X_filled, maes, rmses
        else:
            return X_filled
        

class TDM_RRimputer():
    """
    Round-Robin version of RDM imputer 

    Parameters
    ----------
    models: iterable
        iterable of torch.nn.Module. The j-th model is used to predict the j-th
        variable using all others.

    eps: float, default=0.01
        Sinkhorn regularization parameter.
        
    lr : float, default = 0.01
        Learning rate.

    opt: torch.nn.optim.Optimizer, default=torch.optim.Adam
        Optimizer class to use for fitting.
        
    max_iter : int, default=10
        Maximum number of round-robin cycles for imputation.

    niter : int, default=15
        Number of gradient updates for each model within a cycle.

    batchsize : int, defatul=128
        Size of the batches on which the sinkhorn divergence is evaluated.

    n_pairs : int, default=10
        Number of batch pairs used per gradient update.

    tol : float, default = 0.001
        Tolerance threshold for the stopping criterion.

    weight_decay : float, default = 1e-5
        L2 regularization magnitude.

    order : str, default="random"
        Order in which the variables are imputed.
        Valid values: {"random" or "increasing"}.

    unsymmetrize: bool, default=True
        If True, sample one batch with no missing 
        data in each pair during training.

    scaling: float, default=0.9
        Scaling parameter in Sinkhorn iterations
        c.f. geomloss' doc: "Allows you to specify the trade-off between
        speed (scaling < .4) and accuracy (scaling > .9)"

    """
    def __init__(self,
                 models,
                 projector,
                 rr_lr=1e-2,
                 proj_lr=1e-2,
                 opt=torch.optim.RMSprop,
                 max_niter=10, 
                 niter=15,
                 batchsize=128,
                 n_pairs=10,
                 tol=1e-3,
                 noise=0.1,
                 weight_decay=1e-5,
                 order="random",
                 unsymmetrize=True,
                 scaling=0.9,
                 save_dir_training=None):
        
        self.models = models
        self.projector = projector
        self.rr_lr = rr_lr
        self.proj_lr = proj_lr
        self.opt = opt
        self.max_niter = max_niter
        self.niter = niter
        self.batchsize = batchsize
        self.n_pairs = n_pairs
        self.tol = tol
        self.noise = noise
        self.order = order
        self.unsymmetrize = unsymmetrize
        self.weight_decay = weight_decay
        self.scaling = scaling
        self.save_dir_training = save_dir_training

        self.is_fitted = False


    def fit_transform(self, X, verbose=True, report_interval=1, X_true=None):
       
        X = X.clone()
        n, d = X.shape
        mask = torch.isnan(X).double()
        normalized_tol = self.tol * torch.max(torch.abs(X[~mask.bool()]))

        if self.batchsize > n // 2:
            e = int(np.log2(n // 2))
            self.batchsize = 2**e
            if verbose:
                logging.info(f"Batchsize larger that half size = {len(X) // 2}. Setting batchsize to {self.batchsize}.")


        torch.autograd.set_detect_anomaly(True)


        if torch.sum(mask) < 1.0:
            is_no_missing = True
        else:
            is_no_missing = False

        X_filled = X.detach().clone()   

        if not is_no_missing:
            imps = (self.noise * torch.randn(mask.shape).double() + nanmean(X, 0))[mask.bool()]
            X_filled[mask.bool()] = imps
        
        order_ = torch.argsort(mask.sum(0))
        
        rr_optimizers = [self.opt(self.models[i].parameters(),
                               lr=self.rr_lr, weight_decay=self.weight_decay) for i in range(d)]
        proj_optimizer = self.opt([p for p in self.projector.parameters()], lr=self.proj_lr)

        if X_true is not None:
            maes = np.zeros(self.max_niter)
            rmses = np.zeros(self.max_niter)

        for i in range(self.max_niter):

            if self.order == 'random':
                order_ = np.random.choice(d, d, replace=False)
            X_old = X_filled.detach().clone()

            for l in range(d):
                j = order_[l].item()
                n_not_miss = (~mask[:, j].bool()).sum().item()
                if n - n_not_miss == 0:
                    continue  # no missing value on that coordinate

                for k in range(self.niter):

                    proj_loss = 0
                    rr_loss = 0

                    X_filled = X_filled.detach()
                    X_filled[mask[:, j].bool(), j] = torch.squeeze(self.models[j](X_filled[mask[:, j].bool(), :][:, np.r_[0:j, j+1: d]]))
                    
                    for _ in range(self.n_pairs):

                        idx1 = np.random.choice(n, self.batchsize, replace=False)
                        X1 = X_filled[idx1]

                        if self.unsymmetrize:
                            n_miss = (~mask[:, j].bool()).sum().item()
                            idx2 = np.random.choice(n_miss, self.batchsize, replace= self.batchsize > n_miss)
                            X2 = X_filled[~mask[:, j].bool(), :][idx2]

                        else:
                            idx2 = np.random.choice(n, self.batchsize, replace=False)
                            X2 = X_filled[idx2]

                        X1_p, _ = self.projector(X1)
                        X2_p, _ = self.projector(X2)


                        M_p = torch.cdist(X1_p, X2_p, p=2)

                        a1_p = torch.ones(X1.shape[0]) / X1.shape[0]
                        a2_p = torch.ones(X2.shape[0]) / X2.shape[0]
                        a1_p.requires_grad = False
                        a2_p.requires_grad = False
                        ot_p = ot.emd2(a1_p, a2_p, M_p)


                        rr_loss = rr_loss + ot_p
                        proj_loss = proj_loss + ot_p 

                    if torch.isnan(rr_loss).any() or torch.isinf(rr_loss).any():
                        logging.info("im_loss Nan or inf loss")
                        break

                    if torch.isnan(proj_loss).any() or torch.isinf(proj_loss).any():
                        logging.info("proj_loss Nan or inf loss")
                        break
                    
                    if not is_no_missing:
                        rr_optimizers[j].zero_grad()
                        rr_loss.backward(retain_graph=True)
                        rr_optimizers[j].step()
                    
                    proj_optimizer.zero_grad()
                    proj_loss.backward()
                    proj_optimizer.step()

                #https://discuss.pytorch.org/t/use-of-retain-graph-true/179658/4
                # rr_optimizers[j].zero_grad()
                # rr_loss.backward(retain_graph=False) #allow PyTorch to free the intermediates.
                
                # proj_optimizer.zero_grad()
                # proj_loss.backward()

                
                # Impute with last parameters
                with torch.no_grad():
                    X_filled[mask[:, j].bool(), j] = torch.squeeze(self.models[j](X_filled[mask[:, j].bool(), :][:, np.r_[0:j, j+1: d]]))


            if verbose and (i % report_interval == 0):

                if X_true is not None:
                    maes[i] = MAE(X_filled, X_true, mask).item() 
                    rmses[i] = RMSE(X_filled, X_true, mask).item()

                    logging.info(f'Iteration {i}:\t Imputer Loss: {rr_loss.item():.4f}\t '
                                 f'Projector Loss: {proj_loss.item():.4f}\t '
                                 f'Validation MAE: {maes[i]:.4f}\t'
                                 f'RMSE: {rmses[i]:.4f}')


                else:
                    logging.info(f'Iteration {i}:\t Imputer Loss: {rr_loss.item():.4f}\t '
                                 f'Projector Loss: {proj_loss.item():.4f}\t ')
            
            if torch.norm(X_filled - X_old, p=np.inf) < normalized_tol:
                break
              
        if i == (self.max_niter - 1) and verbose:
            logging.info('Early stopping criterion not reached')

        self.is_fitted = True

        if X_true is not None:
            return X_filled, maes, rmses
        else:
            return X_filled
        

    def transform(self, X, mask, verbose=True, report_interval=1, X_true=None):
        """
        Impute missing values on new data. Assumes models have been previously 
        fitted on other data.
        
        Parameters
        ----------
        X : torch.DoubleTensor or torch.cuda.DoubleTensor, shape (n, d)
            Contains non-missing and missing data at the indices given by the
            "mask" argument. Missing values can be arbitrarily assigned 
            (e.g. with NaNs).

        mask : torch.DoubleTensor or torch.cuda.DoubleTensor, shape (n, d)
            mask[i,j] == 1 if X[i,j] is missing, else mask[i,j] == 0.

        verbose: bool, default=True
            If True, output loss to log during iterations.
            
        report_interval : int, default=1
            Interval between loss reports (if verbose).

        X_true: torch.DoubleTensor or None, default=None
            Ground truth for the missing values. If provided, will output a 
            validation score during training. For debugging only.

        Returns
        -------
        X_filled: numpy array
            Imputed missing data (plus unchanged non-missing data).
       
        """

        assert self.is_fitted, "The model has not been fitted yet."

        n, d = X.shape
        normalized_tol = self.tol * torch.max(torch.abs(X[~mask.bool()]))

        order_ = torch.argsort(mask.sum(0))

        X[mask] = nanmean(X)
        X_filled = X.clone()

        for i in range(self.max_niter):

            if self.order == 'random':
                order_ = np.random.choice(d, d, replace=False)
            X_old = X_filled.clone().detach()

            for l in range(d):

                j = order_[l].item()

                with torch.no_grad():
                    X_filled[mask[:, j].bool(), j] = torch.squeeze(self.models[j](X_filled[mask[:, j].bool(), :][:, np.r_[0:j, j+1: d]]))

            if verbose and (i % report_interval == 0):
                if X_true is not None:
                    logging.info(f'Iteration {i}:\t '
                                 f'Validation MAE: {MAE(X_filled, X_true, mask).item():.4f}\t'
                                 f'RMSE: {RMSE(X_filled, X_true, mask).item():.4f}')

            if torch.norm(X_filled - X_old, p=np.inf) < normalized_tol:
                break

        if i == (self.max_niter - 1) and verbose:
            logging.info('Early stopping criterion not reached')

        return X_filled.detach().cpu().numpy(), MAE(X_filled, X_true, mask).item(), RMSE(X_filled, X_true, mask).item()