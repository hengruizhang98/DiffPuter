import numpy as np
import torch
from torch import nn
from torch.nn.parameter import Parameter

class InterpRealNVP(nn.Module):
    """Normalizing flow model that uses affine coupling layers from RealNVP and a random masking strategy
    Args:
        scaling_nn (torch.nn.Sequential): Neural network architecture for all scaling calculations in affine coupling layer
        translating_nn (torch.nn.Sequential): Neural network architecture for all translating calculations in afficne coupling layer
        mask (list): List of masks to be used specific to each affine coupling transformation
        prior (torch.distributions.MultivariateNormal): The desired distribution for the transformation to the emedding space
    """
    def __init__(self, scaling_nn, translating_nn, mask, prior):
        super(InterpRealNVP, self).__init__()

        self.mask = nn.Parameter(mask, requires_grad=False)
        self.translate_nn = torch.nn.ModuleList([translating_nn() for _ in range(len(mask))])
        self.scale_nn = torch.nn.ModuleList([scaling_nn() for _ in range(len(mask))])
        self.prior = prior


    def forward(self, x):
        # This implements the full transformation from the data space to latent space of the normalizing flow model
        log_det_jac, z = x.new_zeros(x.shape[0]), x
        for index in range(len(self.translate_nn)):
            z, log_det_jac = self.affine_coupling_transform(z, index, log_det_jac)
        return z, log_det_jac


    def inverse(self, z):
        # This implements the full transformation from the latent space to data space
        x = z
        for index in reversed(range(len(self.translate_nn))):
            x = self.inverse_affine_coupling_transform(x, index)
        return x

    def affine_coupling_transform(self, z, index, log_det_jac):
        z_ = self.mask[index] * z
        scale = self.scale_nn[index](z_) * (1-self.mask[index])
        translate = self.translate_nn[index](z_) * (1-self.mask[index])
        z = (1 - self.mask[index]) * (z * torch.exp(scale) + translate)+ z_
        log_det_jac += scale.sum(dim=1)
        return z, log_det_jac

    def inverse_affine_coupling_transform(self, x, index):
        x_ = x*self.mask[index]
        scale = self.scale_nn[index](x_)*(1 - self.mask[index])
        translate = self.translate_nn[index](x_)*(1 - self.mask[index])
        x = x_ + (1 - self.mask[index]) * ((x - translate) * torch.exp(-scale))
        return x


    def log_prob(self,x, args):
        z, logp = self.forward(x)

        log_p = self.prior.log_prob(z.cpu())
        if args.use_cuda:
            log_p = log_p.cuda()
        lgp = log_p + logp

        return z, -lgp.mean()
