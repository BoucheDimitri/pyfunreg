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
from sklearn.decomposition import DictionaryLearning, PCA
from sklearn.gaussian_process.kernels import Matern
from scipy.interpolate import BSpline
import pywt


# Plot parameters
plt.rcParams.update({"pdf.fonttype": 42})
plt.rcParams.update({"font.size": 32})
# plt.rcParams.update({'ps.useafm': True})
plt.rcParams.update({"lines.linewidth": 4})
plt.rcParams.update({"lines.markersize": 10})
plt.rcParams.update({"axes.linewidth": 1})
plt.rcParams.update({"xtick.major.size": 5})
plt.rcParams.update({"xtick.major.width": 1.5})
plt.rcParams.update({"ytick.major.size": 5})
plt.rcParams.update({"ytick.major.width": 1.5})

from kernel.features import RandomFourierFeatures
from regressors.kpl import FeaturesKPLDictselDouble

fontsize = 32
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
ax[0].set_xticks([])
ax[1].set_xticks([])
ax[0].set_yticks([])
ax[0].set_xlabel("$\\theta$")
ax[1].set_xlabel("$\\theta$")
ax[0].set_ylabel("$v(\\theta)$")
ax[1].set_ylabel("$w(\\theta)$")
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
ax[0].set_yticks([])
ax[1].set_yticks([])
ax[0].set_xticks([])
ax[1].set_xticks([])
ax[0].set_xlabel("$\\theta$")
ax[1].set_xlabel("$\\theta$")
ax[0].set_ylabel("$y(\\theta)$")

plt.show()


plt.rcParams.update({"font.size": 40})
fig, ax = plt.subplots(nrows=2, sharex="col")

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
ax[1].set_xlabel("$\\theta$")
# ax[1].set_xlabel("$\\theta$")
ax[0].set_ylabel("$y(\\theta)$")
ax[1].set_ylabel("$y(\\theta)$")
ax[0].set_xticks([])
ax[1].set_xticks([])
ax[0].set_yticks([])
ax[1].set_yticks([])

plt.show()


# ####################### Partial observations lip data #################################################################
viridis = cm.get_cmap('viridis', 12)(np.linspace(0, 1, len(Xlip)))

theta = np.linspace(0, 1, 641)

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
i = 5
Xtrain, Ytrain, Xtest, Ytest, gpdict = load_gp_dataset(seeds_coefs_train[i], seeds_coefs_test[i], return_outdict=True)

n = 10

viridis = cm.get_cmap('viridis', 12)(np.linspace(0, 1, n))

theta = np.linspace(0, 1, 300)

fig, ax = plt.subplots(ncols=3)

for i in range(n):
    ax[0].plot(theta, Xtrain[i], c=viridis[i], alpha=0.9)

for i in range(n):
    ax[1].plot(theta, Ytrain[i], c=viridis[i], alpha=0.9)

for i in range(n):
    inds = np.random.randint(0, 300, 50)
    ax[2].scatter(theta[inds], Ytrain[i][inds], c=viridis[i], alpha=0.9)
lims = ax[1].get_ylim()
ax[2].set_ylim(*lims)
ax[2].set_yticks([])

ax[0].set_title("Input functions $(x_i)_{i=1}^{n}$")
ax[1].set_title("Output functions $(y_i)_{i=1}^{n}$")
ax[2].set_title("Partial output functions $(\\tildey_i)_{i=1}^{n}$")
ax[0].set_xticks([])
ax[1].set_xticks([])
ax[2].set_xticks([])
ax[0].set_yticks([])
ax[1].set_yticks([])
ax[2].set_yticks([])
ax[0].set_xlabel("$\\theta$")
ax[1].set_xlabel("$\\theta$")
ax[2].set_xlabel("$\\theta$")
# ax[0].set_ylabel("$y(\\theta)$")
plt.show()

# ####################### Functional outliers ##################################################################
Xtrain, Ytrain, Xtest, Ytest, gpdict = load_gp_dataset(seeds_coefs_train[0], seeds_coefs_test[0], return_outdict=True)


kersmooth = GaussianKernel(30)
theta = torch.linspace(0, 1, 20).unsqueeze(1)
Kthe = kersmooth(theta)

y = np.random.multivariate_normal(np.zeros(theta.shape[0]), Kthe.numpy())

plt.scatter(theta.squeeze().numpy(), y)
plt.show()


# ###################### Dictionaries ############################################################################
phi, psi, x = pywt.Wavelet("db6").wavefun(level=10)
viridis = cm.get_cmap('viridis', 12)(np.linspace(0, 1, 4))
plt.plot(x, psi, c=viridis[0], alpha=0.9)
plt.plot(x / 2, psi, c=viridis[1], alpha=0.9)
plt.plot(x / 4, psi, c=viridis[2], alpha=0.9)
plt.plot(x / 8, psi, c=viridis[3], alpha=0.9)
plt.ylabel("$\\phi(\\theta)$")
plt.xlabel("$\\theta$")
plt.xticks([])
plt.yticks([])
plt.show()

smoothness = [3, 5, 7]
fig, ax = plt.subplots(ncols=len(smoothness))
for i, s in enumerate(smoothness):
    phi, psi, x = pywt.Wavelet("db" + str(s)).wavefun(level=10)
    viridis = cm.get_cmap('viridis', 12)(np.linspace(0, 1, 3))
    ax[i].plot(x, psi, c=viridis[0], alpha=0.9)
    ax[i].plot(x / 2, psi, c=viridis[1], alpha=0.9)
    ax[i].plot(x / 4, psi, c=viridis[2], alpha=0.9)
    ax[i].set_title("Order " + str(s))
    ax[i].set_xticks([])
    ax[i].set_yticks([])
    ax[i].set_xlabel("$\\theta$")
ax[0].set_ylabel("$\\phi(\\theta)$")
plt.show()


# SPLINES
def knots_generator(domain, n_basis, locs_bounds, width=1, bounds_disc=False, order=3):
        locs = np.linspace(locs_bounds[0], locs_bounds[1], n_basis, endpoint=True)
        pace = width / (order + 1)
        cardinal_knots = np.arange(-width / 2, width / 2 + pace, pace)
        if not bounds_disc:
            knots = [cardinal_knots + loc for loc in locs]
        else:
            knots = []
            for loc in locs:
                knot = cardinal_knots + loc
                knot[knot < domain[0]] = domain[0]
                knot[knot > domain[1]] = domain[1]
                knots.append(knot)
        return knots
viridis = cm.get_cmap('viridis', 12)(np.linspace(0, 1, 9))

fig, ax = plt.subplots(ncols=2)
knots1 = knots_generator([0, 1], 11, [-0.1, 1.1], width=0.25, bounds_disc=True, order=1)
x = np.linspace(0, 1, 300)
for i, knot in enumerate(knots1[1:10]):
    y = BSpline.basis_element(knot, extrapolate=False)(x)
    y[np.isnan(y)] = 0
    ax[0].plot(x, y, c=viridis[i])
ax[0].set_yticks([])
ax[0].set_xticks([])
ax[0].set_xlabel("$\\theta$")
ax[0].set_ylabel("$\\phi(\\theta)$")
ax[0].set_title("Order 2")

dom = [0, 0.7]
knots3 = knots_generator(dom, 11, [dom[0] - 0.1, dom[1] + 0.1], width=0.25, bounds_disc=True, order=3)
x = np.linspace(dom[0], dom[1], 300)
for i, knot in enumerate(knots3[1:10]):
    y = BSpline.basis_element(knot, extrapolate=False)(x)
    y[np.isnan(y)] = 0
    ax[1].plot(x, y, c=viridis[i])
ax[1].set_yticks([])
ax[1].set_xticks([])
ax[1].set_xlabel("$\\theta$")
ax[1].set_title("Order 4")
plt.show()


viridis = cm.get_cmap('viridis', 12)(np.linspace(0, 1, 7))
x = np.linspace(0, 1.5, 300)
y = BSpline.basis_element([0, 0, 0, 0.25, 0.5], extrapolate=False)(x)
y[np.isnan(y)] = 0
plt.plot(x, y, c=viridis[0])
y = BSpline.basis_element([0, 0, 0.25, 0.5, 0.75], extrapolate=False)(x)
y[np.isnan(y)] = 0
plt.plot(x, y, c=viridis[1])
y = BSpline.basis_element([0, 0.25, 0.5, 0.75, 1], extrapolate=False)(x)
y[np.isnan(y)] = 0
plt.plot(x, y, c=viridis[2])
y = BSpline.basis_element([0.25, 0.5, 0.75, 1, 1.25], extrapolate=False)(x)
y[np.isnan(y)] = 0
plt.plot(x, y, c=viridis[3])
y = BSpline.basis_element([0.5, 0.75, 1, 1.25, 1.5], extrapolate=False)(x)
y[np.isnan(y)] = 0
plt.plot(x, y, c=viridis[4])
y = BSpline.basis_element([0.75, 1, 1.25, 1.5, 1.5], extrapolate=False)(x)
y[np.isnan(y)] = 0
plt.plot(x, y, c=viridis[5])
y = BSpline.basis_element([1, 1.25, 1.5, 1.5, 1.5], extrapolate=False)(x)
y[np.isnan(y)] = 0
plt.plot(x, y, c=viridis[6])
plt.ylabel("$\\phi(\\theta)$")
plt.xlabel("$\\theta$")
plt.xticks([])
plt.yticks([])
plt.show()


fig, ax = plt.subplots(ncols=3, sharey="row")
viridis = cm.get_cmap('viridis', 12)(np.linspace(0, 1, 7))

x = np.linspace(0, 1.5, 300)
y = BSpline.basis_element([0, 0, 0, 0.25, 0.5], extrapolate=False)(x)
y[np.isnan(y)] = 0
ax[2].plot(x, y, c=viridis[0])
y = BSpline.basis_element([0, 0, 0.25, 0.5, 0.75], extrapolate=False)(x)
y[np.isnan(y)] = 0
ax[2].plot(x, y, c=viridis[1])
y = BSpline.basis_element([0, 0.25, 0.5, 0.75, 1], extrapolate=False)(x)
y[np.isnan(y)] = 0
ax[2].plot(x, y, c=viridis[2])
y = BSpline.basis_element([0.25, 0.5, 0.75, 1, 1.25], extrapolate=False)(x)
y[np.isnan(y)] = 0
ax[2].plot(x, y, c=viridis[3])
y = BSpline.basis_element([0.5, 0.75, 1, 1.25, 1.5], extrapolate=False)(x)
y[np.isnan(y)] = 0
ax[2].plot(x, y, c=viridis[4])
y = BSpline.basis_element([0.75, 1, 1.25, 1.5, 1.5], extrapolate=False)(x)
y[np.isnan(y)] = 0
ax[2].plot(x, y, c=viridis[5])
y = BSpline.basis_element([1, 1.25, 1.5, 1.5, 1.5], extrapolate=False)(x)
y[np.isnan(y)] = 0
ax[2].plot(x, y, c=viridis[6])

x = np.linspace(0, 1.5, 300)
y = BSpline.basis_element([0, 0, 0.25], extrapolate=False)(x)
y[np.isnan(y)] = 0
ax[0].plot(x, y, c=viridis[1])
y = BSpline.basis_element([0, 0.5, 1], extrapolate=False)(x)
y[np.isnan(y)] = 0
ax[0].plot(x, y, c=viridis[2])
y = BSpline.basis_element([0.25, 0.75, 1.25], extrapolate=False)(x)
y[np.isnan(y)] = 0
ax[0].plot(x, y, c=viridis[3])
y = BSpline.basis_element([0.5, 1, 1.5], extrapolate=False)(x)
y[np.isnan(y)] = 0
ax[0].plot(x, y, c=viridis[4])
y = BSpline.basis_element([0.75, 1.25, 1.5], extrapolate=False)(x)
y[np.isnan(y)] = 0
ax[0].plot(x, y, c=viridis[5])
y = BSpline.basis_element([ 1.25, 1.5, 1.5], extrapolate=False)(x)
y[np.isnan(y)] = 0
ax[0].plot(x, y, c=viridis[6])


x = np.linspace(0, 1.5, 300)
y = BSpline.basis_element([0, 0, 0, 0.3333], extrapolate=False)(x)
y[np.isnan(y)] = 0
ax[1].plot(x, y, c=viridis[0])
y = BSpline.basis_element([0, 0, 0.3333, 0.6666], extrapolate=False)(x)
y[np.isnan(y)] = 0
ax[1].plot(x, y, c=viridis[1])
y = BSpline.basis_element([0, 0.3333, 0.6666, 1], extrapolate=False)(x)
y[np.isnan(y)] = 0
ax[1].plot(x, y, c=viridis[2])
y = BSpline.basis_element([0.3333, 0.6666, 1, 1.3333], extrapolate=False)(x)
y[np.isnan(y)] = 0
ax[1].plot(x, y, c=viridis[3])
y = BSpline.basis_element([0.6666, 1, 1.3333, 1.6666], extrapolate=False)(x)
y[np.isnan(y)] = 0
ax[1].plot(x, y, c=viridis[4])
y = BSpline.basis_element([1, 1.3333, 1.6666, 1.6666], extrapolate=False)(x)
y[np.isnan(y)] = 0
ax[1].plot(x, y, c=viridis[5])
y = BSpline.basis_element([1, 1.6666, 1.6666, 1.6666], extrapolate=False)(x)
y[np.isnan(y)] = 0
ax[1].plot(x, y, c=viridis[6])

plt.show()

plt.ylabel("$\\phi(\\theta)$")
plt.xlabel("$\\theta$")
plt.xticks([])
plt.yticks([])
plt.show()


dl = DictionaryLearning(n_components=5, fit_algorithm="cd", max_iter=5000)
dl.fit(Ytrain - np.expand_dims(Ytrain.mean(0), 0))
viridis = cm.get_cmap('viridis', 12)(np.linspace(0, 1, 7))
plt.plot(dl.components_[0])
plt.plot(dl.components_[1])
plt.plot(dl.components_[2])
plt.plot(dl.components_[3])
plt.show()


fig, ax = plt.subplots(ncols=2)
viridis = cm.get_cmap('viridis', 12)(np.linspace(0, 1, 4))
Ycentered = Ytrain - np.expand_dims(Ytrain.mean(0), 0)
pca = PCA(n_components=4)
pca.fit(Ycentered)
for i in range(4):
    ax[1].plot(pca.components_[i] + Ytrain.mean(0).numpy(), c=viridis[i])
    ax[1].set_yticks([])
    ax[1].set_xticks([])
ax[1].set_xlabel("$\\theta$")
ax[1].set_title("5 first FPCs")
ax[1].set_ylabel("$\\phi(\\theta)$")
viridis = cm.get_cmap('viridis', 12)(np.linspace(0, 1, 50))
for i in range(50):
    ax[0].plot(Ytrain[i], c=viridis[i])
    ax[0].set_yticks([])
    ax[0].set_xticks([])
ax[0].set_title("$(y_i)_{i=1}^n$")
ax[0].set_xlabel("$\\theta$")
ax[0].set_ylabel("$y(\\theta)$")
plt.show()


x = np.linspace(-1, 1, 200)
# plt.plot(x, np.sin(10 * x ** 3) + 2 * np.sin(20 * (x - 0.5) ** 2))
plt.plot(x, np.sin(10 * x ** 3))
plt.plot(x, np.sin(5 * x ** 2 - 5 * x ** 5))
plt.show()



# RKHS
x = np.linspace(0, 1, 200)

nus = [0.2, 0.5, 1.5, 5]

n_fun = 5
funs = [np.zeros((n_fun, len(x))) for i in range(len(nus))]
viridis = cm.get_cmap('viridis', 12)(np.linspace(0, 1, n_fun))
for j in range(len(nus)):
    k = Matern(0.1, nu=nus[j])
    K = k(np.expand_dims(x, 1), np.expand_dims(x, 1))
    for i in range(n_fun):
        alpha = np.random.normal(0, 1, 200)
        funs[j][i] = (K * np.expand_dims(alpha, 1)).sum(0)



fig, ax = plt.subplots(ncols=len(nus), sharey="row")


for j in range(len(nus)):
    for i in range(n_fun):
        funs[j][i] *= 1 / (np.linalg.norm(funs[j][i]))
        ax[j].plot(x, funs[j][i], c=viridis[i])
    ax[j].set_xlabel("$\\theta$")
    ax[j].set_title("$\\nu=" + str(nus[j]) + "$")
    ax[j].set_xticks([])
    ax[j].set_yticks([])
ax[0].set_ylabel("$y(\\theta)$")
plt.show()
