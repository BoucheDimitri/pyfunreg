import os
import sys
from sklearn.decomposition import DictionaryLearning
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from kernel import NystromFeatures, SpeechKernel
from losses import Huber2Loss, SquareLoss
from regressors import FeaturesKPLOtherLoss, FeaturesKPL, SeparableKPL
from optim import acc_proxgd

fontsize = 30
pad = 20

sys.path.append(os.getcwd())
from datasets import load_raw_speech, process_speech

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

dict_learn = DictionaryLearning(n_components=30, alpha=1e-6, tol=1e-5, max_iter=1000, fit_algorithm="cd", random_state=432)
dict_learn.fit(ytrain.numpy())
Yapprox = dict_learn.transform(ytrain.numpy()) @ dict_learn.components_
phi = torch.from_numpy(dict_learn.components_.T)
phi_adj_phi = (1 / phi.shape[0]) * phi.T @ phi
Yproj = (1 / 290) * phi.T @ ytrain.T


kerin = SpeechKernel(gamma=5e-2)
Ktrain = kerin(Xtrain)
Ktest = kerin(Xtrain, Xtest)
m = len(theta)
n_feat = 100
nysfeat = NystromFeatures(kerin, n_feat, 432)
nysfeat.fit(Xtrain, Ktrain)
Z = nysfeat(Xtrain, Ktrain)
n_feat = Z.shape[1]


regu_grid = np.geomspace(1e-9, 1e-5, 30)
qs = [10, 25, 50, 75, 100, 150, 200, 250, 300]
scores = torch.zeros((len(regu_grid), len(qs)))

def compute_scores(preds_full, ypartial):
    sc = 0
    for i in range(len(preds_full)):
        sc += ((preds_full[i, :len(ypartial[0][i])] - ypartial[1][i]) ** 2).sum()
    sc *= (1 / len(preds_full))
    return sc

count_q = 0
for q in qs:
    nysfeat = NystromFeatures(kerin, q, 432)
    nysfeat.fit(Xtrain, Ktrain)
    count_regu = 0
    for regu in regu_grid:
        fpl = FeaturesKPL(regu, nysfeat, phi)
        fpl.fit(Xtrain, ytrain)
        preds_fpl = fpl.predict(Xtest, Ktest)
        scores[count_regu, count_q] = compute_scores(preds_fpl, Ytest)
        count_regu += 1
    count_q += 1
    print(q)

scores_fpl = torch.min(scores, dim=0).values

scores_kpl = []

for regu in regu_grid:
    kpl = SeparableKPL(regu, kerin, torch.eye(phi.shape[1]), phi, phi_adj_phi)
    kpl.fit(Xtrain, ytrain)
    preds_kpl = kpl.predict(Xtest, Ktest)
    scores_kpl.append(compute_scores(preds_kpl, Ytest))

