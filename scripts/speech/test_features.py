import os
import sys
from sklearn.decomposition import DictionaryLearning
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from sklearn.model_selection import KFold
from kernel import NystromFeatures, SpeechKernel
from losses import Huber2Loss, SquareLoss, HuberInfLoss
from regressors import FeaturesKPLOtherLoss, FeaturesKPL, SeparableKPL
from optim import acc_proxgd, AccProxGD
from model_selection import tune_consecutive, cv_consecutive, tune_features, cv_features, product_config

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

n_splits = 5
random_state = 342
kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

# kergamma = torch.logspace(-2, -0.5, 3)
kergamma = torch.logspace(-2, -0.5, 3)
kerins = [SpeechKernel(gamma=gamma) for gamma in kergamma]
feats = [[NystromFeatures(ker, n_feat, 432, thresh=0) for fold in range(n_splits)] for ker in kerins]
Ks = []
for l in range(len(kerins)):
    K = kerins[l](Xtrain)
    Ks.append(K)
    fold = 0
    for train_index, test_index in kf.split(Xtrain):
        Xtr, _ = Xtrain[train_index], Xtrain[test_index]
        Ktr = K[train_index, :][:, train_index]
        feats[l][fold].fit(Xtr, Ktr)
        fold += 1

config = {"regu": torch.logspace(-10, -5, 3), "features": None, "phi": phi, "refit_features": False}
configs = product_config(config, leave_out=["phi"])

estis = [FeaturesKPL(**params) for params in configs]

scs = cv_features(estis[0], feats, Xtrain, ytrain, Ks=Ks, Yeval=Ytrain[1], n_splits=5, reduce_stat="median", random_state=random_state)

best_esti, mses = tune_features(estis, feats, Xtrain, ytrain, Ks, 
                                Yeval=Ytrain[1], n_splits=5, reduce_stat="median", 
                                random_state=342, n_jobs=3)

