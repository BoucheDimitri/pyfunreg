import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colo
from slycot import sb04qd
from scipy import linalg
import pandas as pd
from matplotlib import cm


# Plot parameters
plt.rcParams.update({"pdf.fonttype": 42})
plt.rcParams.update({"font.size": 30})
# plt.rcParams.update({'ps.useafm': True})
plt.rcParams.update({"lines.linewidth": 3.2})
plt.rcParams.update({"lines.markersize": 7})
plt.rcParams.update({"axes.linewidth": 1})
plt.rcParams.update({"xtick.major.size": 5})
plt.rcParams.update({"xtick.major.width": 1.5})
plt.rcParams.update({"ytick.major.size": 5})
plt.rcParams.update({"ytick.major.width": 1.5})

from kernel.features import RandomFourierFeatures
from regressors.kpl import FeaturesKPLDictselDouble

fontsize = 30
pad = 20

sys.path.append(os.getcwd())
from datasets.outliers import add_gp_outliers, add_local_outliers
from datasets import load_gp_dataset, SyntheticGPmixture, load_raw_speech, process_speech
from kernel import GaussianKernel
from kernel import NystromFeatures
from regressors import SeparableKPL, FeaturesKPL, FeaturesKPLDictsel, FeaturesKPLDictselDouble
from optim import acc_proxgd, acc_proxgd_restart



torch.set_default_dtype(torch.float64)

seeds_coefs_train = np.random.choice(np.arange(100, 100000), 10, replace=False)
seeds_coefs_test = np.random.choice(np.arange(100, 100000),10, replace=False)


# ######################## Functional representation #########################################################
nthe = 60
theta = torch.linspace(0, 1, nthe).unsqueeze(1)
y =  - 2 * torch.cos(2 * np.pi * theta) + 2 * torch.cos(2 * np.pi * 2* theta) + torch.cos(2 * np.pi * 3 * theta) -  torch.sin(2 * np.pi * theta) + torch.sin(2 * np.pi * 2 * theta) - torch.sin(2 * np.pi * 3 * theta)

fig, ax = plt.subplots(ncols=2, sharey="row")

locit = [0, 19, 39, 59]
tloc = theta[locit]

ax[1].stem(theta.squeeze().numpy(), y)
ax[0].stem(theta.squeeze().numpy(), 2 * np.random.normal(0, 1, nthe))
ax[0].set_xticks(list(tloc), ["$v(" + str(i + 1) + ")$" for i in locit])
ax[1].set_xticks(list(tloc), ["$w(" + str(i + 1) + ")$" for i in locit])
ax[0].set_yticks([])
plt.show()


# ####################### Examples from speech dataset #########################################################
X, Y = load_raw_speech(os.getcwd() + "/datasets/dataspeech/raw/")
Xtrain, Ytrain_full_ext, Ytrain_full, Xtest, Ytest_full_ext, Ytest_full = process_speech(X, Y, shuffle_seed=None, n_train=300)


# ######################## Lip data outliers #############################################################################
Xlip = pd.read_csv(os.getcwd() + "/datasets/datalip/EMGmatlag.csv", header=None).values.T
Ylip = pd.read_csv(os.getcwd() + "/datasets/datalip/lipmatlag.csv", header=None).values.T

viridis = cm.get_cmap('viridis', 12)(np.linspace(0, 1, len(Xlip)))

for i in range(len(Xlip)):
    plt.plot(Xlip[i], c="tab:blue", alpha=0.5)
plt.show()

Ylip_lout, continds_loc = add_local_outliers(torch.from_numpy(Ylip), intensity=1.5,freq_loc=0.05, freq_sample=0.05, return_inds=True)
Ylip_gout, continds_glob = add_gp_outliers(torch.from_numpy(Ylip), intensity=1.3, freq_sample=0.05, return_inds=True, additive=True)
theta = np.linspace(0, 1, 641)


fig, ax = plt.subplots(ncols=2, sharey="row")

for i in range(len(Ylip)):
    if i not in continds_loc:
        ax[0].plot(theta, Ylip[i], c="tab:blue", alpha=0.3)
for i in continds_loc:
    ax[0].plot(theta, Ylip_lout[i], c="tab:red", alpha=0.9)

for i in range(len(Ylip)):
    if i not in continds_glob:
        ax[1].plot(theta, Ylip[i], c="tab:blue", alpha=0.3)
for i in continds_glob:
    ax[1].plot(theta, Ylip_gout[i], c="tab:red", alpha=0.9)
ax[0].set_xlabel("$\\theta$")
ax[1].set_xlabel("$\\theta$")

plt.show()

# ####################### Partial observations lip data #################################################################
viridis = cm.get_cmap('viridis', 12)(np.linspace(0, 1, len(Xlip)))

fig, ax = plt.subplots(ncols=3, sharey="col")

for i in range(len(Xlip)):
    ax[0].plot(theta, Xlip[i], c=viridis[i], alpha=0.9)

for i in range(len(Xlip)):
    inds = np.random.randint(0, 641, 100)
    ax[1].plot(theta, Ylip[i], c=viridis[i], alpha=0.9)

for i in range(len(Xlip)):
    inds = np.random.randint(0, 641, 50)
    ax[2].scatter(theta[inds], Ylip[i][inds], c=viridis[i], alpha=0.9)
plt.show()

# ####################### Partial observations toy data ##############################################################
# Load dataset
Xtrain, Ytrain, Xtest, Ytest, gpdict = load_gp_dataset(seeds_coefs_train[0], seeds_coefs_test[0], return_outdict=True)

# ####################### Functional outliers ##################################################################
Xtrain, Ytrain, Xtest, Ytest, gpdict = load_gp_dataset(seeds_coefs_train[0], seeds_coefs_test[0], return_outdict=True)


kersmooth = GaussianKernel(30)
theta = torch.linspace(0, 1, 20).unsqueeze(1)
Kthe = kersmooth(theta)

y = np.random.multivariate_normal(np.zeros(theta.shape[0]), Kthe.numpy())

plt.scatter(theta.squeeze().numpy(), y)
plt.show()