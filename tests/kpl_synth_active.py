import os
import sys
from unicodedata import bidirectional
import torch
import numpy as np
import matplotlib.pyplot as plt
from slycot import sb04qd

fontsize = 30
pad = 20

sys.path.append(os.getcwd())
from datasets.outliers import add_gp_outliers
from datasets import add_gp_outliers, load_gp_dataset, SyntheticGPmixture
from kernel import GaussianKernel
from regressors import SeparableKPL, FeaturesKPLOtherLoss, FeaturesKPLWorking
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
# CORRUPT_GLOBAL_PARAMS = {"freq_sample":0.1, "intensity": (0., 4., 10), "seed_gps": 56}


STDS_GPS_OUT = [0.05, 0.1, 0.5, 0.7]
GP_SCALE = 1.5

atoms_stds = [0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.5, 0.7]
n_per_std = 40
# Parasite atoms
stds_out = []
for n, _ in enumerate(atoms_stds):
    stds_out += [atoms_stds[n] for i in range(n_per_std)]
stds_in = stds_out
scale = 1.5
n_atoms = len(stds_in)
gamma_cov = torch.Tensor([stds_in, stds_out]).numpy()
data_gp = SyntheticGPmixture(n_atoms=n_atoms, gamma_cov=gamma_cov, scale=scale)
data_gp.drawGP(theta, seed_gp=764)
big_dict = data_gp.GP_output
phi = big_dict.T
m = len(theta)
gram_mat = (1 / m) * phi.T @ phi
phi *= torch.sqrt((1 / torch.diag(gram_mat).unsqueeze(0)))
phi_adj_phi = (1 / m) * phi.T @ phi
d = phi.shape[1]

Xtrain, Ytrain, Xtest, Ytest = Xtrain.numpy(), Ytrain.numpy(), Xtest.numpy(), Ytest.numpy()
kerin = GaussianKernel(0.01)
Ktrain = kerin(Xtrain)

m = len(theta)
n_feat = 100
nysfeat = NystromFeatures(kerin, n_feat, 432)
nysfeat.fit(Xtrain, Ktrain)
accproxgd = AccProxGD(n_epoch=20000, stepsize0=1, tol=1e-6, acc_temper=20)

wkpl = FeaturesKPLWorking(1e-8, 1.5e-3, nysfeat, phi.numpy(), accproxgd, phi_adj_phi.numpy())
monitor = wkpl.fit(Xtrain, Ytrain, Ktrain)
preds = wkpl.predict(Xtest)
mse = ((preds - Ytest) ** 2).mean()
print(mse)
alpha0 = wkpl.alpha.copy()


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
