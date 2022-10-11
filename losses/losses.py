import torch
import numpy as np


def huber(y, kappa):
    mask_sup = y.abs() > kappa
    mask_inf = torch.logical_not(mask_sup)
    z = torch.zeros(y.shape)
    z[mask_sup] = kappa * (y[mask_sup].abs() - 0.5 * kappa)
    z[mask_inf] = 0.5 * y[mask_inf] ** 2
    return z

def huber_grad(y, kappa):
    mask_sup = y.abs() > kappa
    mask_inf = torch.logical_not(mask_sup)
    z = torch.zeros(y.shape)
    z[mask_sup] = kappa * y[mask_sup].sign()
    z[mask_inf] = y[mask_inf]
    return z



class Huber2Loss:

    def __init__(self, loss_param):
        self.loss_param = loss_param

    def __call__(self, y):
        norms = (1 / np.sqrt(y.shape[1])) * torch.norm(y, p=2, dim=1)
        return huber(norms, self.loss_param).mean()

    def grad(self, y):
        ynorms = (1 / np.sqrt(y.shape[1])) * torch.norm(y, p=2, dim=1)
        g = huber_grad(ynorms, self.loss_param)
        G = (y / ynorms.unsqueeze(1)) * g.unsqueeze(1)
        return (1 / len(ynorms)) * G


class HuberInfLoss:

    def __init__(self, loss_param):
        self.loss_param = loss_param
    
    def __call__(self, y):
        return huber(y, self.loss_param).mean()

    def grad(self, y):
        return (1 / len(y)) * huber_grad(y, self.loss_param)
    

class SquareLoss:

    def __init__(self):
        pass

    def __call__(self, y):
        norms = ((1 / np.sqrt(y.shape[1])) * torch.norm(y, p=2, dim=1)) ** 2
        return 0.5 * norms.mean()
    
    def grad(self, y):
        return (1 / len(y)) * y

                
