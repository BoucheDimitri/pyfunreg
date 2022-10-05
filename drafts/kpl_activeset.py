import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from slycot import sb04qd
from scipy import linalg

from kernel.features import RandomFourierFeatures
from regressors.kpl import FeaturesKPLDictselDouble

fontsize = 30
pad = 20

sys.path.append(os.getcwd())
from datasets.outliers import add_gp_outliers
from datasets import load_gp_dataset, SyntheticGPmixture
from kernel import GaussianKernel
from kernel import NystromFeatures
from regressors import SeparableKPL, FeaturesKPL, FeaturesKPLDictsel, FeaturesKPLDictselDouble
from optim import acc_proxgd, acc_proxgd_restart



torch.set_default_dtype(torch.float64)

seeds_coefs_train = np.random.choice(np.arange(100, 100000), 10, replace=False)
seeds_coefs_test = np.random.choice(np.arange(100, 100000),10, replace=False)
theta = torch.linspace(0, 1, 100)


# ############################ EXAMPLES WITH OUTLIERS ##############################################
Xtrain, Ytrain, Xtest, Ytest, gpdict = load_gp_dataset(seeds_coefs_train[0], seeds_coefs_test[0], return_outdict=True)

kerin = GaussianKernel(0.01)
Ktrain = kerin(Xtrain)
phi = gpdict.T
# Normalize atoms
m = len(theta)
gram_mat = (1 / m) * phi.T @ phi
phi *= torch.sqrt((1 / torch.diag(gram_mat).unsqueeze(0)))
phi_adj_phi = (1 / m) * phi.T @ phi

d = phi.shape[1]

n_feat = 100
nysfeat = NystromFeatures(kerin, n_feat, 432)
nysfeat.fit(Xtrain, Ktrain)

nyskpl = FeaturesKPLDictsel(1e-8, nysfeat, phi, phi_adj_phi)
nyskpl.fit(Xtrain, Ytrain, Ktrain, tol=1e-4, d=20, beta=0.8)


Ktest = kerin(Xtrain, Xtest)
preds = nyskpl.predict(Xtest, Ktest)
((preds - Ytest) ** 2).mean()




def bst_matrix(alpha, tau):
    norm = (alpha**2).sum(1).sqrt()
    mask_st = torch.where(norm >= tau)
    mask_ze = torch.where(norm < tau)
    alpha[mask_st] = alpha[mask_st] - alpha[mask_st] / \
        norm[mask_st].reshape((-1, 1)) * tau
    alpha[mask_ze] = 0
    return(alpha)


# nyskpl = FeaturesKPLDictsel(1e-5, nysfeat, phi, phi_adj_phi)
# nyskpl.fit(Xtrain, Ytrain, Ktrain, tol=1e-4, d=10, beta=0.8)
# Ktest = kerin(Xtrain, Xtest)
# preds = nyskpl.predict(Xtest, Ktest)
# ((preds - Ytest) ** 2).mean()

# scores = []

# for tol in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
#     nyskpl.fit(Xtrain, Ytrain, Ktrain, tol=tol)
#     preds = nyskpl.predict(Xtest, Ktest)
#     scores.append(((preds - Ytest) ** 2).mean())

# PARASITE ATOMS
stds_in = [0.005, 0.005, 0.005, 0.005]
stds_out = [0.005, 0.005, 0.005, 0.005]
scale = 1.5
n_atoms = len(stds_in)
gamma_cov = torch.Tensor([stds_in, stds_out]).numpy()
data_gp = SyntheticGPmixture(n_atoms=n_atoms, gamma_cov=gamma_cov, scale=scale)
data_gp.drawGP(theta, seed_gp=764)
parasites = data_gp.GP_output

dictpara = torch.cat((parasites, gpdict))
kerin = GaussianKernel(0.01)
Ktrain = kerin(Xtrain)
phi = dictpara.T
# Normalize atoms
m = len(theta)
gram_mat = (1 / m) * phi.T @ phi
phi *= torch.sqrt((1 / torch.diag(gram_mat).unsqueeze(0)))
phi_adj_phi = (1 / m) * phi.T @ phi

d = phi.shape[1]

m = len(theta)
n_feat = 100
nysfeat = NystromFeatures(kerin, n_feat, 432)
nysfeat.fit(Xtrain, Ktrain)


nyskpl = FeaturesKPLDictsel(1e-5, nysfeat, phi, phi_adj_phi)
nyskpl.fit(Xtrain, Ytrain, Ktrain, tol=1e-4, d=20, beta=0.8)

alpha0 = torch.normal(0, 1, (d, n_feat))
step_size = 1
alpha, monitored = acc_proxgd(alpha0, nyskpl.prox, nyskpl.obj, nyskpl.grad, step_size, n_epoch=20000, tol=1e-6, beta=0.8, d=20, monitor=None)

Ktest = kerin(Xtrain, Xtest)
preds = nyskpl.predict(Xtest, Ktest)
((preds - Ytest) ** 2).mean()

# TODO: On a pas pris la norme fonctionnelle dans le prox alors que l'on a pris la norme fonctionnelle pour phi_adj_phi et Y_proj, le problème vient peut être de là

#
def acc_proxgd(alpha0, prox, obj, grad, step_size, n_epoch=20000, tol=1e-6, beta=0.8, d=20, monitor=None):
    alpha_minus1 = alpha0
    alpha_minus2 = alpha0
    step_size = 1
    converged = False
    monitored = []
    for epoch in range(0, n_epoch):
        acc_cste = epoch / (epoch + 1 + d)
        alpha_v = alpha_minus1 + acc_cste * (alpha_minus1 - alpha_minus2)
        grad_v = grad(alpha_v)
        # step_size = acc_proxgd_lsearch(
        #     prox, obj, step_size, alpha_v, grad_v, beta)
        alpha = prox(alpha_v - step_size * grad_v, step_size)
        if monitor is not None:
            monitored.append(monitor(alpha))
        if alpha_minus1.norm() < 1e-10:
            raise ValueError("Norm too small")
        diff = (alpha - alpha_minus1).norm() / alpha_minus1.norm()
        print(diff)
        if diff < tol:
            converged = True
            break
        alpha_minus2 = alpha_minus1.detach().clone()
        alpha_minus1 = alpha.detach().clone()
    return alpha, monitored


# Plain GD
gamma = 1
alpha = torch.normal(0, 1, (d, n_feat))
for i in range(20000): 
    alpha -= gamma * (nyskpl.grad(alpha) + nyskpl.regu * alpha)
    print(nyskpl.obj(alpha) + nyskpl.regu * (alpha ** 2).sum())


nyskpl = FeaturesKPLDictsel(1e-4, nysfeat, phi, phi_adj_phi)
nyskpl.fit(Xtrain, Ytrain, Ktrain, tol=1e-4, d=20, beta=0.8)

crits = []
scores = []
# Prox GD
gamma0 = 1
discount = 0.8
alpha = torch.normal(0, 1, (d, n_feat))
alpha0 = torch.normal(0, 1, (d, n_feat))
for i in range(1, 10000):
    objcurr = nyskpl.obj(alpha)
    gradcurr = nyskpl.grad(alpha)
    stop = False
    gamma = gamma0
    while not stop:
        # alphaplus = nyskpl.prox(beta - gamma * nyskpl.grad(beta), gamma)
        alphaplus = bst_matrix(alpha - gamma * gradcurr, gamma * nyskpl.regu)
        objplus = nyskpl.obj(alphaplus)
        alphadiff = alphaplus - alpha
        if objplus > objcurr + (gradcurr * alphadiff).sum() + (1 / (2 * gamma)) * (alphadiff ** 2).sum():
            gamma *= discount
            # print("Discounted")
        else:
            stop = True
            alpha0 = alpha
            alpha = alphaplus
    stop = False
    # # No prox
    # alpha2 = beta - gamma * (nyskpl.grad(beta) + nyskpl.regu * beta)
    crit = torch.abs(nyskpl.full_obj(alpha) - nyskpl.full_obj(alpha0)) / torch.abs(nyskpl.full_obj(alpha0))
    print(crit)
    crits.append(crit)
    scores.append(nyskpl.full_obj(alpha))





crits = []
scores = []
# Accelerated GD
gamma = 0.1
discount = 0.8
alpha0 = torch.normal(0, 1, (d, n_feat))
alpha1 = torch.normal(0, 1, (d, n_feat))
alpha2 = torch.normal(0, 1, (d, n_feat))
beta = torch.normal(0, 1, (d, n_feat))
acc_temper = 20
for epoch in range(10000):
    # acc_cste = ((i - 1) / (i + 2))
    acc_cste = epoch / (epoch + 1 + acc_temper)
    beta = alpha1 + acc_cste * (alpha1 - alpha0)
    # alpha2 = nyskpl.prox(beta - gamma * nyskpl.grad(beta), gamma)
    objbeta = nyskpl.obj(beta)
    gradbeta = nyskpl.grad(beta)
    stop = False
    while not stop:
        # alphaplus = nyskpl.prox(beta - gamma * nyskpl.grad(beta), gamma)
        alphaplus = bst_matrix(beta - gamma * nyskpl.grad(beta), gamma * nyskpl.regu)
        objplus = nyskpl.obj(alphaplus)
        if objplus > objbeta + (gradbeta * (alphaplus - beta)).sum() + (1 / (2 * gamma)) * ((alphaplus - beta) ** 2).sum():
            gamma *= discount
            print("Discounted")
        else:
            stop = True
            alpha2 = alphaplus
    stop = False
    # # No prox
    # alpha2 = beta - gamma * (nyskpl.grad(beta) + nyskpl.regu * beta)
    crit = torch.abs(nyskpl.full_obj(alpha2) - nyskpl.full_obj(alpha1)) / torch.abs(nyskpl.full_obj(alpha1))
    print(crit)
    crits.append(crit)
    scores.append(nyskpl.full_obj(alpha2))
    # print(crit)
    alpha0 = alpha1
    alpha1 = alpha2
    # print(nyskpl.obj(alpha2))


m = len(theta)
n_feat = 100
nysfeat = NystromFeatures(kerin, n_feat, 432)
nysfeat.fit(Xtrain, Ktrain)
# nyskpl = FeaturesKPLDictsel(1e-2, nysfeat, phi, phi_adj_phi)
# nyskpl.fit(Xtrain, Ytrain, Ktrain, tol=1e-4, acc_temper=20, beta=0.8, stepsize0=0.7)
nyskpl = FeaturesKPLDictselDouble(0, 1e-6, nysfeat, phi, phi_adj_phi)
nyskpl.fit(Xtrain, Ytrain, Ktrain, tol=1e-6, acc_temper=20, beta=0.8, stepsize0=1)

Ktest = kerin(Xtrain, Xtest)
preds = nyskpl.predict(Xtest, Ktest)
pred_coefs = nyskpl.predict_coefs(Xtest, Ktest)
((preds - Ytest) ** 2).mean()

plt.plot(preds[0])
plt.plot(Ytest[0])
plt.show()

predcore = (phi[:, 4:] @ pred_coefs[4:, :]).T

plt.plot(predcore[0])
plt.plot(Ytest[0])
plt.show()

l = 2
prednoise = (phi[:, l].unsqueeze(1) @ pred_coefs[l, :].unsqueeze(0)).T
plt.plot(prednoise[0])
plt.plot(Ytest[0])
plt.show()


prednoise = (phi[:, :4] @ pred_coefs[:4, :].unsqueeze(0)).T
plt.plot(prednoise[0])
plt.plot(Ytest[0])
plt.show()




crits = []
scores = []
# Accelerated GD
gamma = 1e-5
discount = 0.8
alpha0 = torch.normal(0, 1, (d, n_feat))
alpha1 = torch.normal(0, 1, (d, n_feat))
alpha2 = torch.normal(0, 1, (d, n_feat))
beta = torch.normal(0, 1, (d, n_feat))
for i in range(1, 10000):
    beta = alpha1 + ((i - 1) / (i + 2)) * (alpha1 - alpha0)
    # alpha2 = nyskpl.prox(beta - gamma * nyskpl.grad(beta), gamma)
    alpha2 = nyskpl.prox(beta - gamma * nyskpl.grad(beta), gamma)
    # # No prox
    # alpha2 = beta - gamma * (nyskpl.grad(beta) + nyskpl.regu * beta)
    crit = ((alpha2 - alpha1) ** 2).sum().sqrt() / (alpha1 ** 2).sum().sqrt()
    crits.append(crit)
    scores.append(nyskpl.obj(alpha2))
    # print(crit)
    # print(nyskpl.obj(alpha2) + nyskpl.regu * torch.sqrt((alpha2 ** 2).sum(dim=1)).sum())
    alpha0 = alpha1
    alpha1 = alpha2
    print(nyskpl.obj(alpha2))
