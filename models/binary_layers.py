import torch
import torch.nn as nn
from torch.autograd import *


class BinaryLayer(Function):

    def __init__(self):
        pass

    def forward(self, x):
        return torch.sign(x)

    def backward(self, grad_output):
        return grad_output


class BinaryLayer2(nn.Module):
    """Binary layer as defined in the paper"""
    def forward(self, x):
        probs_tensor = torch.rand(x.size())
        errors = Variable(torch.FloatTensor(x.size()))
        probs_threshold = torch.div(torch.add(x, 1), 2)
        alpha = 1-x[probs_tensor <= probs_threshold.data]
        beta = -x[probs_tensor > probs_threshold.data] - 1
        errors[probs_tensor <= probs_threshold.data] = alpha
        errors[probs_tensor > probs_threshold.data] = beta
        y = x + errors
        return y

    @staticmethod
    def backward(grad_output):
        return grad_output
