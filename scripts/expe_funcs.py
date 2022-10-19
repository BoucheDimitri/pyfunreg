import os
import sys
from sklearn.decomposition import DictionaryLearning
import torch
import numpy as np
from sklearn.model_selection import KFold
import pathlib




exec_path = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(str(exec_path.parent.parent))
sys.path.append(str(exec_path.parent))
sys.path.append(str(exec_path))




from kernel import NystromFeatures, SpeechKernel
from losses import Huber2Loss, SquareLoss, HuberInfLoss
from regressors import FeaturesKPLOtherLoss, FeaturesKPL, SeparableKPL
from optim import acc_proxgd, AccProxGD
from model_selection import tune_consecutive, cv_consecutive, tune_features, cv_features, product_config
from datasets import load_raw_speech, process_speech

fontsize = 30
pad = 20

sys.path.append(os.getcwd())
from datasets import load_raw_speech, process_speech


def draw_seeds(n_averaging, seed):
    np.random.seed(seed)
    seeds_coefs_train = np.random.choice(np.arange(100, 100000), n_averaging, replace=False)
    seeds_coefs_test = np.random.choice(np.arange(100, 100000), n_averaging, replace=False)
    seeds_corrupt = np.random.choice(np.arange(100, 100000), n_averaging, replace=False)
    seeds_cv = np.random.choice(np.arange(100, 100000), n_averaging, replace=False)
    return seeds_coefs_train, seeds_coefs_test, seeds_corrupt, seeds_cv


def load_speech_dataset(seed, path, n_train=300):
    X, Y = load_raw_speech(path)
    Xtrain, Ytrain_full_ext, Ytrain_full, Xtest, Ytest_full_ext, Ytest_full = process_speech(
        X, Y, shuffle_seed=seed, n_train=n_train)
    return Xtrain, Ytrain_full_ext, Ytrain_full, Xtest, Ytest_full_ext, Ytest_full


def pretrain_nystrom_features(X, n_feat, kernel, gammas, cv_seed, nys_seed, n_splits=5):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=cv_seed)
    kernels = [kernel(gamma=gamma) for gamma in gammas]
    feats = [[NystromFeatures(ker, n_feat, nys_seed, thresh=0) for s in range(n_splits)] for ker in kernels]
    Ks = []
    for l in range(len(kernels)):
        K = kernels[l](X)
        Ks.append(K)
        s = 0
        for train_index, test_index in cv.split(X):
            Xtr, _ = X[train_index], X[test_index]
            Ktr = K[train_index, :][:, train_index]
            feats[l][s].fit(Xtr, Ktr)
            s += 1
    return Ks, feats


def pretrain_dictionaries(Y, cv_seed, n_splits=5, n_components=30, alpha=1e-6, tol=1e-5, max_iter=1000, dl_seed=674):
    phis = []
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=cv_seed)
    for train_index, test_index in cv.split(Y):
        dict_learn = DictionaryLearning(n_components=n_components, alpha=alpha, tol=tol, max_iter=max_iter, fit_algorithm="cd", random_state=dl_seed)
        dict_learn.fit(Y[train_index].numpy())
        phi = torch.from_numpy(dict_learn.components_.T)
        phis.append(phi)
    return phis

# seed = 1454
# n_train = 300

# X, Y = load_raw_speech(str(os.getcwd()) + "/datasets/dataspeech/raw/")
# Xtrain, Ytrain_full_ext, Ytrain_full, Xtest, Ytest_full_ext, Ytest_full = process_speech(X, Y, shuffle_seed=seed, n_train=n_train)
# key = "LA"
# Ytrain_ext, Ytrain, Ytest_ext, Ytest \
#     = Ytrain_full_ext[key], Ytrain_full[key], Ytest_full_ext[key], Ytest_full[key]

# theta = Ytrain[0][0]
# ytrain = torch.Tensor(Ytrain_ext[1])
# ytest_ext = torch.Tensor(Ytest_ext[1])
# Xtrain = torch.Tensor(Xtrain)
# Xtest = torch.Tensor(Xtest)

