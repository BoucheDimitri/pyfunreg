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
from datasets import load_gp_dataset, SyntheticGPmixture
from kernel import GaussianKernel
from kernel import NystromFeatures
from regressors import FeaturesKPL, FeaturesKPLWorking

# Plot parameters
plt.rcParams.update({"pdf.fonttype": 42})
plt.rcParams.update({"font.size": 30})
# plt.rcParams.update({'ps.useafm': True})
plt.rcParams.update({"lines.linewidth": 5})
plt.rcParams.update({"lines.markersize": 10})
plt.rcParams.update({"axes.linewidth": 1.5})
plt.rcParams.update({"xtick.major.size": 5})
plt.rcParams.update({"xtick.major.width": 1.5})
plt.rcParams.update({"ytick.major.size": 5})
plt.rcParams.update({"ytick.major.width": 1.5})



torch.set_default_dtype(torch.float64)

N_AVG = 10

seeds_coefs_train = np.random.choice(np.arange(100, 100000), N_AVG, replace=False)
seeds_coefs_test = np.random.choice(np.arange(100, 100000), N_AVG, replace=False)
theta = torch.linspace(0, 1, 100)

# regus12 = torch.logspace(-7, -5, 100)
regus12 = torch.Tensor(np.geomspace(4e-7, 1e-6, 50))
regus = torch.logspace(-8, -3, 50)

scores12 = []
recos12 = []
scores = []


for t in range(N_AVG):

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

    # Fit methods with 12 regularization and parasite dictionary
    score12 = []
    reco12 = []
    good_set = [31, 32, 33, 34]
    for regu12 in regus12:
        nyskpl = FeaturesKPLWorking(0, regu12, nysfeat, phi, phi_adj_phi)
        nyskpl.fit(Xtrain, Ytrain, Ktrain, tol=1e-7, acc_temper=20, beta=0.8, stepsize0=0.7)
        preds = nyskpl.predict(Xtest, Ktest)
        noise_atoms = [i for i in nyskpl.working if i not in good_set]
        reco12.append(len(noise_atoms))
        score12.append(((preds - Ytest) ** 2).mean().item())
    scores12.append(score12)
    recos12.append(reco12)

    # Fit method with 2 regularization and no parasites
    dictnopara = gpdict
    phi = dictnopara.T
    # Normalize atoms
    m = len(theta)
    gram_mat = (1 / m) * phi.T @ phi
    phi *= torch.sqrt((1 / torch.diag(gram_mat).unsqueeze(0)))
    phi_adj_phi = (1 / m) * phi.T @ phi
    d = phi.shape[1]
    score = []
    for regu in regus:
        nyskpl = FeaturesKPL(regu, nysfeat, phi, phi_adj_phi)
        nyskpl.fit(Xtrain, Ytrain, Ktrain)
        preds = nyskpl.predict(Xtest, Ktest)
        score.append(((preds - Ytest) ** 2).mean().item())
    scores.append(score)

    print(t)

scs_mean =torch.min(torch.Tensor(scores), dim=1).values.mean()



fig, ax = plt.subplots(nrows=2, sharex="col")
scs12_mean = torch.Tensor(scores12).mean(dim=0)
scs12_std = torch.Tensor(scores12).std(dim=0)
recos12_mean = torch.Tensor(recos12).mean(dim=0)
recos12_std = torch.Tensor(recos12).std(dim=0)
ax[0].plot(regus12, scs12_mean, marker="o")
ax[0].set_xscale("log")
ax[0].set_yscale("log")
ax[0].set_ylabel("NMSE")
ax[1].plot(regus12, recos12_mean, marker="o")
ax[1].set_xscale("log")
ax[1].set_ylabel("# Non-essential atoms")
ax[1].set_xlabel("$\\lambda$")
ax[0].hlines(scs_mean, xmin = regus12.min(), xmax = regus12.max(), linestyle="dashed", color="k", label="Best FPL with true dictionary")
ax[0].legend()
plt.show()