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
from regressors import SeparableKPL, FeaturesKPL, FeaturesKPLDictsel, FeaturesKPLDictselDouble, FeaturesKPLWorking
from optim import acc_proxgd, acc_proxgd_restart



torch.set_default_dtype(torch.float64)

seeds_coefs_train = np.random.choice(np.arange(100, 100000), 10, replace=False)
seeds_coefs_test = np.random.choice(np.arange(100, 100000),10, replace=False)
theta = torch.linspace(0, 1, 100)


# Load dataset
Xtrain, Ytrain, Xtest, Ytest, gpdict = load_gp_dataset(seeds_coefs_train[0], seeds_coefs_test[0], return_outdict=True)

# Parasite atoms
stds_out = [0.005, 0.005, 0.005, 0.005, 0.0075, 0.0075, 0.0075, 0.0075, 0.01, 0.01, 0.01, 0.01]
stds_in = stds_out
scale = 1.5
n_atoms = len(stds_in)
gamma_cov = torch.Tensor([stds_in, stds_out]).numpy()
data_gp = SyntheticGPmixture(n_atoms=n_atoms, gamma_cov=gamma_cov, scale=scale)
data_gp.drawGP(theta, seed_gp=764)
parasites_gp = data_gp.GP_output
sp_atoms = []
knots = [np.array([0, 0.05, 0.1, 0.15, 0.2]) + a for a in np.arange(0, 0.95, 0.05)]
for knot in knots:
    b = BSpline.basis_element(knot, extrapolate=False)
    beval = b(theta.numpy())
    beval[np.isnan(beval)] = 0
    sp_atoms.append(beval)
parasites_sp = torch.from_numpy(np.array(sp_atoms))
dictpara = torch.cat((parasites_gp, parasites_sp, gpdict))

# Projection operators with normalized atoms
phi = dictpara.T
m = len(theta)
gram_mat = (1 / m) * phi.T @ phi
phi *= torch.sqrt((1 / torch.diag(gram_mat).unsqueeze(0)))
phi_adj_phi = (1 / m) * phi.T @ phi
d = phi.shape[1]
m = len(theta)

# Features
n_feat = 100
kerin = GaussianKernel(0.01)
Ktrain = kerin(Xtrain)
nysfeat = NystromFeatures(kerin, n_feat, 432)
nysfeat.fit(Xtrain, Ktrain)
Ktest = kerin(Xtrain, Xtest)

kerin = GaussianKernel(0.01)
Ktrain = kerin(Xtrain)
nysfeat = NystromFeatures(kerin, n_feat, 432)
nysfeat.fit(Xtrain, Ktrain)

nyskpl = FeaturesKPLWorking(0, 6.8e-7, nysfeat, phi, phi_adj_phi)
nyskpl.fit(Xtrain, Ytrain, Ktrain, tol=1e-6, acc_temper=20, beta=0.8, stepsize0=0.7)


nyskpl = FeaturesKPLDictselDouble(0, 1e-6, nysfeat, phi, phi_adj_phi)
nyskpl.fit(Xtrain, Ytrain, Ktrain, tol=1e-6, acc_temper=20, beta=0.8, stepsize0=0.4)

Ktest = kerin(Xtrain, Xtest)
preds = nyskpl.predict(Xtest, Ktest)
pred_coefs = nyskpl.predict_coefs(Xtest, Ktest)
((preds - Ytest) ** 2).mean().item()

plt.plot(preds[0])
plt.plot(Ytest[0])
plt.show()

clakpl = FeaturesKPL(1e-10, nysfeat, phi, phi_adj_phi=None, center_out=False)
clakpl.fit(Xtrain, Ytrain, Ktrain)

preds = clakpl.predict(Xtest, Ktest)
((preds - Ytest) ** 2).mean().item()



dictnopara = gpdict
phi = dictnopara.T
# Normalize atoms
m = len(theta)
gram_mat = (1 / m) * phi.T @ phi
phi *= torch.sqrt((1 / torch.diag(gram_mat).unsqueeze(0)))
phi_adj_phi = (1 / m) * phi.T @ phi
d = phi.shape[1]
m = len(theta)
n_feat = 200

clakpl = SeparableKPL(1e-10, kerin, torch.eye(d), phi, phi_adj_phi=None, center_out=False)
clakpl.fit(Xtrain, Ytrain, Ktrain)

preds = clakpl.predict(Xtest, Ktest)
((preds - Ytest) ** 2).mean().item()