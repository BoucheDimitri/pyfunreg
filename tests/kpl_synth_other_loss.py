import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from slycot import sb04qd

fontsize = 30
pad = 20

sys.path.append(os.getcwd())
from datasets.outliers import add_gp_outliers
from datasets import add_gp_outliers, load_gp_dataset
from kernel import GaussianKernel
from regressors import SeparableKPL, FeaturesKPLOtherLoss
from functional_data import FourierBasis
from kernel import NystromFeatures
from losses import Huber2Loss
from optim import AccProxGD


torch.set_default_dtype(torch.float64)

seeds_coefs_train = np.random.choice(np.arange(100, 100000), 10, replace=False)
seeds_coefs_test = np.random.choice(np.arange(100, 100000),10, replace=False)
theta = torch.linspace(0, 1, 100)


# ############################ EXAMPLES WITH OUTLIERS ##############################################
Xtrain, Ytrain, Xtest, Ytest, gpdict = load_gp_dataset(seeds_coefs_train[0], seeds_coefs_test[0], return_outdict=True)
Ytrain_corr, _ = add_gp_outliers(Ytrain, Xeval=None, freq_sample=0.1, intensity=4, seed=789, seed_gps=56)
# CORRUPT_GLOBAL_PARAMS = {"freq_sample":0.1, "intensity": (0., 4., 10), "seed_gps": 56}

Xtrain, Ytrain, Xtest, Ytest = Xtrain.numpy(), Ytrain.numpy(), Xtest.numpy(), Ytest.numpy()
Ytrain_corr = Ytrain_corr.numpy()
kerin = GaussianKernel(0.01)
Ktrain = kerin(Xtrain)

fourdict = FourierBasis(0, 40, (0, 1))

phi = fourdict.compute_matrix(theta.numpy())

# Normalize atoms
m = len(theta)
phi_adj_phi = np.eye(fourdict.n_basis)


lbda_grid = np.geomspace(1e-7, 1e-5, 10)
loss_params = np.flip(np.linspace(0.01, 0.1, 20))


m = len(theta)
n_feat = 100
nysfeat = NystromFeatures(kerin, n_feat, 432)
nysfeat.fit(Xtrain, Ktrain)
hubloss = Huber2Loss(0.1)
accproxgd = AccProxGD(n_epoch=20000, stepsize0=1, tol=1e-6, acc_temper=20)

hubkpl = FeaturesKPLOtherLoss(1e-7, hubloss, nysfeat, phi, accproxgd)
monitor = hubkpl.fit(Xtrain, Ytrain_corr, Ktrain)
preds = hubkpl.predict(Xtest)
mse = ((preds - Ytest) ** 2).mean()
alpha0 = hubkpl.alpha.copy()


hubloss = Huber2Loss(0.09526316)
hubkpl = FeaturesKPLOtherLoss(1e-7, hubloss, nysfeat, phi, accproxgd)
monitor = hubkpl.fit(Xtrain, Ytrain, Ktrain, alpha0=alpha0)
preds = hubkpl.predict(Xtest)
mse = ((preds - Ytest) ** 2).mean()
alpha0 = hubkpl.alpha.copy()





scores = []
for regu in torch.logspace(-10, 1, 100):
    sepkpl = SeparableKPL(regu, kerin, torch.eye(phi.shape[1]), phi)
    sepkpl.fit(Xtrain, Ytrain)
    preds = sepkpl.predict(Xtest)
    mse = ((preds - Ytest) ** 2).mean()
    scores.append(mse)
