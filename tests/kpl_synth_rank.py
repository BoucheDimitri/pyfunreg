import os
import sys
from unicodedata import bidirectional
import torch
import numpy as np
import matplotlib.pyplot as plt
from slycot import sb04qd
import numpy as np
import time

fontsize = 30
pad = 20

sys.path.append(os.getcwd())
from datasets.outliers import add_gp_outliers
from datasets import add_gp_outliers, load_gp_dataset, SyntheticGPmixture
from kernel import GaussianKernel
from regressors import SeparableKPL, FeaturesKPLOtherLoss, FeaturesKPLWorking, FeaturesKPL
from functional_data import FourierBasis
from kernel import NystromFeatures
from losses import Huber2Loss
from optim import AccProxGD



torch.set_default_dtype(torch.float64)

seeds_coefs_train = np.random.choice(np.arange(100, 100000), 10, replace=False)
seeds_coefs_test = np.random.choice(np.arange(100, 100000),10, replace=False)
theta = torch.linspace(0, 1, 300)


# ############################ EXAMPLES WITH OUTLIERS ##############################################
Xtrain, Ytrain, Xtest, Ytest, gpdict = load_gp_dataset(seeds_coefs_train[0], seeds_coefs_test[0], return_outdict=True)
# CORRUPT_GLOBAL_PARAMS = {"freq_sample":0.1, "intensity": (0., 4., 10), "seed_gps": 56}


Xtrain, Ytrain, Xtest, Ytest = Xtrain.numpy(), Ytrain.numpy(), Xtest.numpy(), Ytest.numpy()


STDS_GPS_OUT = [0.05, 0.1, 0.5, 0.7]
GP_SCALE = 1.5

ns_per_std = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 40, 60, 80, 100]
eigvals = []

for i in ns_per_std:
    atoms_stds = [0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.5, 0.7]
    n_per_std = i
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
    u, V = np.linalg.eigh(phi_adj_phi)
    eigvals.append(u)
    print(i)

thresh = 1e-4
n_taken = []
for u in eigvals:
    n_taken.append(len(u[u > thresh * len(u)]))


n_taken_fix = []
thresh = 1e-1
for u in eigvals:
    n_taken_fix.append(len(u[u > thresh]))


fact = 1e-3
for 

plt.plot(u)
plt.yscale("log")
plt.show()




kerin = GaussianKernel(0.01)
Ktrain = kerin(Xtrain)

m = len(theta)
n_feat = 100
nysfeat = NystromFeatures(kerin, n_feat, 432)
nysfeat.fit(Xtrain, Ktrain)


kpl = FeaturesKPL(1e-7, nysfeat, phi.numpy(), phi_adj_phi.numpy())
start = time.process_time()
kpl.fit(Xtrain, Ytrain, Ktrain)
end = time.process_time()
print(end - start)
preds = kpl.predict(Xtest)
mse = ((preds - Ytest) ** 2).mean()
print(mse)


thresh = 1e-1
uthresh = u[u > thresh]
Vthresh = V[:, u > thresh]
phi_thresh = phi @ Vthresh
phi_adj_phi_thresh = np.diag(uthresh)

tkpl = FeaturesKPL(1e-10, nysfeat, phi_thresh.numpy(), phi_adj_phi_thresh)
start = time.process_time()
tkpl.fit(Xtrain, Ytrain, Ktrain)
end = time.process_time()
print(end - start)
preds = tkpl.predict(Xtest)
mse = ((preds - Ytest) ** 2).mean()
print(mse)

lbda_grid = np.geomspace(1e-7, 1e-2, 200)
scs_thresh = []
for lbda in lbda_grid:
    tkpl = FeaturesKPL(lbda, nysfeat, phi_thresh.numpy(), phi_adj_phi_thresh)
    tkpl.fit(Xtrain, Ytrain, Ktrain)
    preds = tkpl.predict(Xtest)
    mse = ((preds - Ytest) ** 2).mean()
    scs_thresh.append(mse)

