from abc import ABC
import numpy as np
import time
import torch


def check_inputs(X, Y=None):
    if not isinstance(X, torch.Tensor):
        Xin = torch.from_numpy(X)
        if Y is not None:
            Yin = torch.from_numpy(Y)
    else:
        Xin = X
    if Y is None:
        Yin = Xin.detach().clone()
    else:
        Yin = Y
    return Xin, Yin


class GaussianKernel:

    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, X, Y=None):
        Xin, Yin = check_inputs(X, Y)
        dists = torch.cdist(Xin, Yin)
        return torch.exp(- self.gamma * dists ** 2)


class LaplaceKernel:

    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, X, Y=None):
        Xin, Yin = check_inputs(X, Y)
        dists = torch.cdist(Xin, Yin, p=1)
        return torch.exp(- self.gamma * dists)


class SpeechKernel:

    def __init__(self, gamma=1, center=True, reduce=True):
        self.gamma = gamma
        self.center = center
        self.reduce = reduce

    def __call__(self, X, Y=None):
        Xin, Yin = check_inputs(X, Y)
        n = len(Xin)
        m = len(Yin)
        # Since Y is always Xtrain, this normalization only uses the training data
        if self.reduce:
            mfcc_stds = torch.std(Yin, dim=(0, 1)).unsqueeze(0).unsqueeze(0)
        else:
            mfcc_stds = torch.ones(1, 1, Yin.shape[-1])
        if self.center:
            mfcc_means = torch.mean(Yin, dim=(0, 1)).unsqueeze(0).unsqueeze(0)
        else:
            mfcc_means = torch.zeros(1, 1, Yin.shape[-1])
        K = torch.zeros((m, n))
        Xcentered, Ycentered = (Xin - mfcc_means) / mfcc_stds, (Yin - mfcc_means) / mfcc_stds
        for j in range(m):
            exp_dist = torch.exp(- self.gamma * torch.sum((Xcentered - Ycentered[j]) ** 2, dim=2))
            K[j, :] = torch.mean(exp_dist, dim=1)
        return K.T


class KAMKernel:

    def __init__(self, kernel_in, kernel_evals, space, domain, normalize=False):
        self.kernel_in = kernel_in
        self.kernel_eval = kernel_evals
        self.Klocs_in = self.kernel_in(np.expand_dims(space, axis=1), np.expand_dims(space, axis=1))
        self.domain = domain

    def __call__(self, x0, x1, return_cpu_time=False):
        start = time.process_time()
        if x0.shape == x1.shape:
            if np.allclose(x0, x1):
                K = self.call_train(x0)
            else:
                K = self.call_test(x0, x1)
        else:
            K = self.call_test(x0, x1)
        end = time.process_time()
        if return_cpu_time:
            return K, end - start
        else:
            return K

    def call_train(self, x):
        n = len(x)
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                Kevals = self.kernel_eval(x[i], x[j])
                inter_const = self.domain[0, 1] - self.domain[0, 0]
                K[i, j] = inter_const ** 2 * np.mean(Kevals * self.Klocs_in)
                K[j, i] = inter_const ** 2 * np.mean(Kevals.T * self.Klocs_in)
            print(i)
        return K

    def call_test(self, x0, x1):
        m, n = len(x0), len(x1)
        K = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                Kevals = self.kernel_eval(x0[i], x1[j])
                inter_const = self.domain[0, 1] - self.domain[0, 0]
                K[i, j] = inter_const ** 2 * np.mean(Kevals * self.Klocs_in)
            print(i)
        return K