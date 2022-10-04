import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from slycot import sb04qd
from scipy import linalg

from kernel.features import RandomFourierFeatures

fontsize = 30
pad = 20

sys.path.append(os.getcwd())
from datasets.outliers import add_gp_outliers
from datasets import load_gp_dataset, SyntheticGPmixture
from kernel import GaussianKernel
from kernel import NystromFeatures
from regressors import SeparableKPL, FeaturesKPL, FeaturesKPLDictsel
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

m = len(theta)
n_feat = 100
nysfeat = NystromFeatures(kerin, n_feat, 432)
nysfeat.fit(Xtrain, Ktrain)


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
stds_in = [0.01, 0.01, 0.03, 0.03, 0.3, 0.3]
stds_out = [0.01, 0.01, 0.03, 0.03, 0.3, 0.3]
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


nyskpl = FeaturesKPLDictsel(1e-8, nysfeat, phi, phi_adj_phi)
nyskpl.fit(Xtrain, Ytrain, Ktrain, tol=1e-4, d=20, beta=0.8)
Ktest = kerin(Xtrain, Xtest)
preds = nyskpl.predict(Xtest, Ktest)
((preds - Ytest) ** 2).mean()