import torch


def huber(y, kappa):
    mask_sup = y.abs() > kappa
    mask_inf = torch.logical_not(mask_sup)
    z = torch.zeros(y.shape)
    z[mask_sup] = kappa * (y[mask_sup].abs() - 0.5 * kappa)
    z[mask_inf] = 0.5 * y[mask_inf] ** 2
    return z

def huber_grad(y, kappa, return_masks=False):
    mask_sup = y.abs() > kappa
    mask_inf = torch.logical_not(mask_sup)
    z = torch.zeros(y.shape)
    z[mask_sup] = kappa * y[mask_sup].sign()
    z[mask_inf] = y[mask_inf]
    if return_masks:
        return z, mask_inf, mask_sup
    else:
        return z



class Huber2Loss:

    def __init__(self, loss_param):
        self.loss_param = loss_param

    def __call__(self, y):
        norms = (1 / y.shape[1]) * torch.norm(y, p=2, dim=1)
        return huber(norms, self.loss_param).mean()

    def grad(self, y):
        ynorms = (1 / y.shape[1]) * torch.norm(y, p=2, dim=1)
        g, mask_inf, mask_sup = huber_grad(ynorms, self.loss_param, return_masks=True)
        g = g.unsqueeze(1)
        z = torch.zeros(y.shape)
        z[mask_inf, :] = y[mask_inf, :]
        z[mask_sup, :] = (1 / ynorms[mask_sup].unsqueeze(1)) * y[mask_sup, :]
        return (1 / len(ynorms)) * z * g

                
