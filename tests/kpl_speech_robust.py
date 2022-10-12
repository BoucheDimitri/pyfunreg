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
import time

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
n_feat = 300
nysfeat = NystromFeatures(kerin, n_feat, 432)
nysfeat.fit(Xtrain, Ktrain)
Z = nysfeat(Xtrain, Ktrain)
n_feat = Z.shape[1]

accproxgd = AccProxGD(n_epoch=20000, stepsize0=1e3, tol=1e-5, acc_temper=20)

# hloss = Huber2Loss(40)
# hloss = Huber2Loss(0.005)
hloss = HuberInfLoss(0.005)
nyskpl = FeaturesKPLOtherLoss(1e-9, hloss, nysfeat, phi, accproxgd, sylvester_init=True)
monitor = nyskpl.fit(Xtrain, ytrain, Ktrain)

nyskplinit = FeaturesKPL(0.5e-9, nysfeat, phi)
nyskplinit.fit(Xtrain, ytrain, Ktrain)

start = time.process_time()
hloss = HuberInfLoss(0.003)
nyskpl2 = FeaturesKPLOtherLoss(1e-9, hloss, nysfeat, phi, accproxgd)
monitor = nyskpl2.fit(Xtrain, ytrain, Ktrain)
end = time.process_time()





nyskplinit = FeaturesKPL(1e-9)

preds = nyskpl2.predict(Xtest, Ktest)
compute_scores(preds, Ytest).item()

i=0
plt.plot(Ytest[0][i], preds[i, :len(Ytest[0][i])])
plt.plot(Ytest[0][i], Ytest[1][i])
plt.show()