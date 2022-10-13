import os
import sys
from sklearn.decomposition import DictionaryLearning
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from kernel import NystromFeatures, SpeechKernel
from losses import Huber2Loss, SquareLoss, HuberInfLoss
from regressors import FeaturesKPLOtherLoss, FeaturesKPL, SeparableKPL
from optim import acc_proxgd, AccProxGD
from model_selection import tune_consecutive, cv_consecutive, product_config

fontsize = 30
pad = 20

sys.path.append(os.getcwd())
from datasets import load_raw_speech, process_speech

def compute_scores(preds_full, ypartial):
    sc = 0
    for i in range(len(preds_full)):
        sc += ((preds_full[i, :len(ypartial[0][i])] - ypartial[1][i]) ** 2).sum()
    sc *= (1 / len(preds_full))
    return sc

seed = 1454
n_train = 300

X, Y = load_raw_speech(str(os.getcwd()) + "/datasets/dataspeech/raw/")
Xtrain, Ytrain_full_ext, Ytrain_full, Xtest, Ytest_full_ext, Ytest_full = process_speech(X, Y, shuffle_seed=seed, n_train=n_train)
key = "LA"
Ytrain_ext, Ytrain, Ytest_ext, Ytest \
    = Ytrain_full_ext[key], Ytrain_full[key], Ytest_full_ext[key], Ytest_full[key]

theta = Ytrain[0][0]
ytrain = torch.Tensor(Ytrain_ext[1])
ytest_ext = torch.Tensor(Ytest_ext[1])
Xtrain = torch.Tensor(Xtrain)
Xtest = torch.Tensor(Xtest)

# Dictionary
dict_learn = DictionaryLearning(n_components=30, alpha=1e-6, tol=1e-5, max_iter=1000, fit_algorithm="cd", random_state=432)
dict_learn.fit(ytrain.numpy())
Yapprox = dict_learn.transform(ytrain.numpy()) @ dict_learn.components_
phi = torch.from_numpy(dict_learn.components_.T)
phi_adj_phi = (1 / phi.shape[0]) * phi.T @ phi
Yproj = (1 / 290) * phi.T @ ytrain.T

# Nystrom features
kerin = SpeechKernel(gamma=5e-2)
Ktrain = kerin(Xtrain)
Ktest = kerin(Xtrain, Xtest)
m = len(theta)
n_feat = 300
nysfeat = NystromFeatures(kerin, n_feat, 432, thresh=0)
nysfeat.fit(Xtrain, Ktrain)
Z = nysfeat(Xtrain, Ktrain)
n_feat = Z.shape[1]

# Optimizer
accproxgd = AccProxGD(n_epoch=20000, stepsize0=1e3, tol=1e-5, acc_temper=20)

# hloss = Huber2Loss(40)
# hloss = Huber2Loss(0.005)
losses = [HuberInfLoss(0.01), HuberInfLoss(0.005), HuberInfLoss(0.001)]

config = {"loss": None, "regu": torch.logspace(-10, -5, 2), "features": nysfeat, "optimizer": accproxgd, "phi": phi, "refit_features": True}
configs = product_config(config, leave_out=["phi"])

estis = [FeaturesKPLOtherLoss(**params) for params in configs]

scs = cv_consecutive(estis[0], losses, Xtrain, ytrain, K=Ktrain, Yeval=Ytrain[1], n_splits=5, reduce_stat="median", random_state=342)

best_esti, mses = tune_consecutive(estis, losses, Xtrain, ytrain, K=None, 
                     Yeval=Ytrain[1], n_splits=5, reduce_stat="median", 
                     random_state=342, n_jobs=3)

