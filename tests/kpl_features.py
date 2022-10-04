import os
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
from datasets import load_gp_dataset
from kernel import GaussianKernel
from kernel import NystromFeatures
from regressors import SeparableKPL, FeaturesKPL


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

nyskpl = FeaturesKPL(1e-7, nysfeat, phi, phi_adj_phi)
nyskpl.fit(Xtrain, Ytrain, Ktrain)
Ktest = kerin(Xtrain, Xtest)
preds = nyskpl.predict(Xtest, Ktest)

plt.plot(preds[0])
plt.plot(Ytest[0])
plt.show()

