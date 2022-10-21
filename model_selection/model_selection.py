from inspect import Attribute
from typing import Iterable
import torch
import numpy as np
from sklearn.model_selection import KFold
import os
from joblib import delayed, Parallel, parallel_backend
from multiprocessing import cpu_count

torch.set_default_dtype(torch.float64)


def compute_mse(preds_full, Ypartial):
    sc = 0
    for i in range(len(preds_full)):
        sc += ((preds_full[i, :len(Ypartial[i])] - Ypartial[i]) ** 2).sum()
    sc *= (1 / len(preds_full))
    return sc


def cv_consecutive(esti, losses, X, Y, K=None, Yeval=None, n_splits=5, reduce_stat="median", random_state=342):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    mses = torch.zeros((5, len(losses)))
    count = 0
    for train_index, test_index in kf.split(X):
        Xtrain, Xtest = X[train_index], X[test_index]
        Ytrain, Ytest = Y[train_index], Y[test_index]
        if K is not None:
            Ktrain = K[train_index, :][:, train_index]
            Ktest = K[train_index, :][:, test_index]
        else:
            Ktrain, Ktest = None, None
        for l, loss in enumerate(losses):
            try:
                alpha0 = esti.alpha.detach().clone()
            except AttributeError:
                alpha0 = None
            esti.set_loss(loss)
            esti.fit(Xtrain, Ytrain, Ktrain, alpha0)
            preds = esti.predict(Xtest, Ktest)
            if Yeval is not None:
                mses[count, l] = compute_mse(preds, [Yeval[j] for j in test_index])
            else:
                mses[count, l] = compute_mse(preds, Ytest)
        # print(count)
        # Reinitialize alpha for next fold
        esti.alpha = None
        count += 1
    if reduce_stat == "mean":
        return torch.tensor(mses).mean(dim=0).clone().detach()
    else:
        return torch.tensor(mses).quantile(0.5, dim=0).clone().detach()


def cv_features(esti, features, X, Y, Ks=None, Yeval=None, phis=None, n_splits=5, reduce_stat="median", random_state=342):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    mses = torch.zeros((5, len(features)))
    count = 0
    for train_index, test_index in kf.split(X):
        Xtrain, Xtest = X[train_index], X[test_index]
        Ytrain, Ytest = Y[train_index], Y[test_index]
        if phis is not None:
            esti.set_phi(phis[count])
        for l, feats in enumerate(features):
            if Ks is not None:
                Ktrain = Ks[l][train_index, :][:, train_index]
                Ktest = Ks[l][train_index, :][:, test_index]
            else:
                Ktest = None
                Ktrain = None
            esti.set_features(feats[count])
            esti.fit(Xtrain, Ytrain, Ktrain)
            preds = esti.predict(Xtest, Ktest)
            if Yeval is not None:
                mses[count, l] = compute_mse(preds, [Yeval[j] for j in test_index])
            else:
                mses[count, l] = compute_mse(preds, Ytest)
        # print(count)
        count += 1
    if reduce_stat == "mean":
        return torch.tensor(mses).mean(dim=0).clone().detach()
    else:
        return torch.tensor(mses).quantile(0.5, dim=0).clone().detach()


def tune_consecutive(estis, losses, X, Y, K=None, 
                     Yeval=None, n_splits=5, reduce_stat="median", 
                     random_state=342, n_jobs=-1):
    # with parallel_backend("loky"):
    #     mses = Parallel(n_jobs=n_jobs)(
    #         delayed(cv_consecutive)(esti, losses, X, Y, K, Yeval, n_splits, reduce_stat, random_state) 
    #         for esti in estis)
    mses = [cv_consecutive(esti, losses, X, Y, K, Yeval, n_splits, reduce_stat, random_state) for esti in estis]
    mses = torch.stack(mses)
    esti_argmin, losses_argmin = mses.argmin() // len(losses), mses.argmin() % len(losses)
    best_esti = estis[esti_argmin]
    best_esti.set_loss(losses[losses_argmin])
    best_esti.fit(X, Y, K)
    return best_esti, mses


def tune_features(estis, features, X, Y, Ks=None, 
                  Yeval=None, phis=None, phi_test=None, n_splits=5, reduce_stat="median", 
                  random_state=342, n_jobs=-1):
    # with parallel_backend("loky"):
    #     mses = Parallel(n_jobs=n_jobs)(
    #         delayed(cv_features)(esti, features, X, Y, Ks, Yeval, phis, n_splits, reduce_stat, random_state) 
    #         for esti in estis)
    mses = [cv_features(esti, features, X, Y, Ks, Yeval, phis, n_splits, reduce_stat, random_state) for esti in estis]
    mses = torch.stack(mses)
    esti_argmin, feats_argmin = mses.argmin() // len(features), mses.argmin() % len(features)
    best_esti = estis[esti_argmin]
    best_esti.set_features(features[feats_argmin][0])
    if phi_test is not None:
        best_esti.set_phi(phi_test)
    best_esti.fit(X, Y, Ks[feats_argmin], refit_features=True)
    return best_esti, mses


def test_esti_partial(esti, X, Ypartial):
    preds = esti.predict(X)
    return compute_mse(preds, Ypartial)
