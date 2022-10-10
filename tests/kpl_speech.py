import os
import sys
from sklearn.decomposition import DictionaryLearning
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from kernel import NystromFeatures, SpeechKernel
from losses import Huber2Loss
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
Ytrain1 = torch.Tensor(Ytrain_ext[1])
Xtrain = torch.Tensor(Xtrain)

dict_learn = DictionaryLearning(n_components=30, alpha=1e-6, tol=1e-5, max_iter=1000, fit_algorithm="cd", random_state=432)
dict_learn.fit(Ytrain1.numpy())
Yapprox = dict_learn.transform(Ytrain1.numpy()) @ dict_learn.components_
phi = torch.from_numpy(dict_learn.components_.T)
phi_adj_phi = (1 / phi.shape[0]) * phi.T @ phi
Yproj = (1 / 290) * phi.T @ Ytrain1.T


kerin = SpeechKernel(gamma=5e-2)
Ktrain = kerin(Xtrain)
m = len(theta)
n_feat = 100
nysfeat = NystromFeatures(kerin, n_feat, 432)
nysfeat.fit(Xtrain, Ktrain)
Z = nysfeat(Xtrain, Ktrain)
n_feat = Z.shape[1]

hloss = Huber2Loss(40)

nyskpl = FeaturesKPLOtherLoss(2e-8, hloss, nysfeat, phi)

monitor = nyskpl.fit(Xtrain, Ytrain1, Ktrain, stepsize0=1e10, monitor=nyskpl.obj, tol=1e-20)


alpha = torch.normal(0, 1, (phi.shape[1], nysfeat.n_features_eff))
alpha, monitor = acc_proxgd(alpha, nyskpl.prox, nyskpl.obj, nyskpl.full_obj, nyskpl.grad, n_epoch=10000, monitor=nyskpl.obj, stepsize0=1000)


grad_v = nyskpl.grad(alpha)
t = accgd_lsearch(nyskpl.obj, 1000, alpha, grad_v, beta=0.8)


gsqr = (2 / 300) * (phi_adj_phi @ alpha @ Z.T @ Z - Yproj @ Z)

gsqr2 = ((2 / 300) * (Z @ alpha.T @ phi.T - Ytrain1) @ (phi / phi.shape[0])).T @ Z


def accgd_lsearch(obj, t0, alpha_v, grad_v, beta=0.2):
    t = t0
    stop = False
    while not stop:
        alpha_plus = alpha_v - t * grad_v
        term1 = obj(alpha_plus)
        term21 = obj(alpha_v)
        term22 = (grad_v * (alpha_plus - alpha_v)).sum()
        term23 = 0.5 * (1 / t) * ((alpha_plus - alpha_v) ** 2).sum()
        term2 = term21 + term22 + term23
        if term1 > term2:
            t *= beta
            print("red")
        else:
            stop = True
    return t


def acc_proxgd(alpha0, prox, obj, obj_full, grad, n_epoch=20000, tol=1e-6, beta=0.8, acc_temper=20, monitor=None, stepsize0=0.1):
    alpha_minus1 = alpha0
    alpha_minus2 = alpha0
    step_size = stepsize0
    converged = False
    monitored = []
    for epoch in range(0, n_epoch):
        acc_cste = epoch / (epoch + 1 + acc_temper)
        alpha_v = alpha_minus1 + acc_cste * (alpha_minus1 - alpha_minus2)
        grad_v = grad(alpha_v)
        step_size = acc_proxgd_lsearch(
            prox, obj, step_size, alpha_v, grad_v, beta)
        alpha = prox(alpha_v - step_size * grad_v, step_size)
        if monitor is not None:
            monitored.append(monitor(alpha))
        # if alpha_minus1.norm() < 1e-10:
        #     raise ValueError("Norm too small")
        # diff = (alpha - alpha_minus1).norm() / alpha_minus1.norm()
        # diff = (obj_full(alpha) - obj_full(alpha_minus1)).abs()
        # print(diff)
        # if diff < tol:
        #     converged = True
        #     break
        print(obj_full(alpha).item())
        alpha_minus2 = alpha_minus1.detach().clone()
        alpha_minus1 = alpha.detach().clone()
    return alpha, monitored



gamma = 10
scores = []
scores.append(nyskpl.obj(alpha))

for i in range(10000):
    alpha -= gamma * nyskpl.grad(alpha)
    scores.append(nyskpl.obj(alpha))
    print(scores[-1].item())


nyskpl.alpha = alpha
preds = nyskpl.predict(torch.Tensor(Xtrain), Ktrain)

plt.plot(preds[0])
plt.plot(Ytrain1[0])
plt.show()


nyskpl2 = FeaturesKPL(1e-8, nysfeat, phi)
nyskpl2.fit(Xtrain, Ytrain1, Ktrain)
preds2 = nyskpl2.predict(Xtrain)

plt.plot(preds2[0], label="Pred")
plt.plot(Ytrain1[0], label="True")
plt.legend()
plt.show()
