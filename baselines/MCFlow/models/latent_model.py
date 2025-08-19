import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np

'''Regresses a new latent vector which represents a more likely sample in the embedding'''

class LatentToLatentApprox(nn.Module):
    """Feed foward neural network that regresses new embeddings in the latent space
    Args:
        input_dim (int): Dimensionality of the input.
        num_hidden_neurons (list): Number of neurons to use for each linear layer.
    """

    def __init__(self, input_dim, num_hidden_neurons):
        super(LatentToLatentApprox, self).__init__()

        self.h_0 = nn.Linear(input_dim, num_hidden_neurons[0])
        self.h_1 = nn.Linear(num_hidden_neurons[0], num_hidden_neurons[1])
        self.h_2 = nn.Linear(num_hidden_neurons[1], num_hidden_neurons[2])
        self.h_3 = nn.Linear(num_hidden_neurons[2], num_hidden_neurons[3])
        self.h_4 = nn.Linear(num_hidden_neurons[3], num_hidden_neurons[4])
        self.act = nn.LeakyReLU()

    def forward(self, z):

        z = self.act(self.h_0(z))
        z = self.act(self.h_1(z))
        z = self.act(self.h_2(z))
        z = self.act(self.h_3(z))

        return self.h_4(z)
