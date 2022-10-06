import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from slycot import sb04qd
from scipy import linalg
from scipy.interpolate import BSpline

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

Ktest = kerin(Xtrain, Xtest)

featkpl = FeaturesKPL(1e-8, nysfeat, phi, phi_adj_phi)
featkpl.fit(Xtrain, Ytrain, Ktrain)
preds = featkpl.predict(Xtest, Ktest)
pred_coefs = featkpl.predict_coefs(Xtest, Ktest)
((preds - Ytest) ** 2).mean()

plt.plot(preds[0])
plt.plot(Ytest[0])
plt.show()

l = 2
prednoise = (phi[:, l].unsqueeze(1) @ pred_coefs[l, :].unsqueeze(0)).T
plt.plot(prednoise[0])
plt.plot(Ytest[0])
plt.show()


# nyskpl = FeaturesKPLDictsel(1e-2, nysfeat, phi, phi_adj_phi)
# nyskpl.fit(Xtrain, Ytrain, Ktrain, tol=1e-4, acc_temper=20, beta=0.8, stepsize0=0.7)
nyskpl = FeaturesKPLDictselDouble(0, 1e-6, nysfeat, phi, phi_adj_phi)
nyskpl.fit(Xtrain, Ytrain, Ktrain, tol=1e-7, acc_temper=20, beta=0.8, stepsize0=1, n_epoch=30000)

Ktest = kerin(Xtrain, Xtest)
preds = nyskpl.predict(Xtest, Ktest)
pred_coefs = nyskpl.predict_coefs(Xtest, Ktest)
((preds - Ytest) ** 2).mean()

i = 5
plt.plot(preds[i])
plt.plot(Ytest[i])
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
