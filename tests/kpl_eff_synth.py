import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from slycot import sb04qd
from scipy import linalg

from kernel.features import RandomFourierFeatures

fontsize = 30
pad = 20

sys.path.append(os.getcwd())
from datasets.outliers import add_gp_outliers
from datasets import load_gp_dataset
from kernel import GaussianKernel
from kernel import NystromFeatures
from regressors import SeparableKPL


torch.set_default_dtype(torch.float64)

seeds_coefs_train = np.random.choice(np.arange(100, 100000), 10, replace=False)
seeds_coefs_test = np.random.choice(np.arange(100, 100000),10, replace=False)
theta = torch.linspace(0, 1, 100)


# ############################ EXAMPLES WITH OUTLIERS ##############################################
Xtrain, Ytrain, Xtest, Ytest, gpdict = load_gp_dataset(seeds_coefs_train[0], seeds_coefs_test[0], return_outdict=True)

kerin = GaussianKernel(0.01)
Ktrain = kerin(Xtrain)
phi = gpdict.T
# Normalize atoms
m = len(theta)
gram_mat = (1 / m) * phi.T @ phi
phi *= torch.sqrt((1 / torch.diag(gram_mat).unsqueeze(0)))
phi_adj_phi = (1 / m) * phi.T @ phi

d = phi.shape[1]

m = len(theta)
n_feat = 100
nysfeat = NystromFeatures(kerin, n_feat, 432)
nysfeat.fit(Xtrain, Ktrain)
Ztrain = nysfeat(Xtrain, Ktrain)
regu = 1e-10

n = len(Xtrain)

Yproj = (1 / m) * phi.T @ Ytrain.T
alpha = sb04qd(n_feat, d, (Ztrain.T @ Ztrain).numpy() / (regu * n), phi_adj_phi.numpy(), Ztrain.T.numpy() @ Yproj.T.numpy() / (regu * n))
alpha = torch.from_numpy(alpha.T)
preds = Ztrain @ alpha.T @ phi.T

plt.plot(preds[0])
plt.plot(Ytrain[0])
plt.show()



u, U = linalg.eigh((Ztrain.T @ Ztrain).numpy())
v, V = linalg.eigh()

B = V.T @ Ztrain.T.numpy() @ Yproj.T.numpy()

A = np.zeros((d, n_feat))
for r in range(d):
    A[r] = (B[r] / u[r]) @ V @ regu * n / u[r]


plt.plot(preds[0])
plt.plot(Ytrain[0])
plt.show()



class SeparableKPL:
    """
    Parameters
    ----------
    kernel : callable
        Must support being called on two array_like objects X0, X1. If len(X0) = n_samples0 and len(X1) = n_samples1,
        must returns a torch.Tensor object with shape [n_samples1, n_samples0].
    B : torch.Tensor
        Matrix encoding the similarities between output tasks, of shape [n_atoms, n_atoms]
    phi : torch.Tensor
        Discretized projection operator, of shape [n_locations, n_atoms]
    regu : float
        Regularization parameter
    center_out : bool
        Should output functions be centered
    """
    def __init__(self, regu, kernel, B, phi, phi_adj_phi=None, center_out=False):

        super().__init__()
        self.kernel = kernel
        self.regu = regu
        self.alpha = None
        self.X = None
        self.phi = phi
        self.B = B
        self.phi_adj_phi = phi_adj_phi
        self.center_out= center_out
        self.Ymean = None

    def forget_phi(self):
        self.phi = None

    def set_phi(self, phi):
        self.phi = phi

    def fit(self, X, Y, K=None):
        """
        Parameters
        ----------
        X : array_like
            Input data, of shape [n_samples, n_features]
        Y : array_like
            Output functions, of shape [n_samples, n_locations]
        """
        # Memorize training input data
        self.X = X
        if K is None:
            K = self.kernel(X, X)
        n = K.shape[0]
        m = Y.shape[1]
        if self.center_out:
            self.Ymean = Y.mean(dim=0)
            Ycenter = Y - self.Ymean
        else:
            Ycenter = Y
        Yproj = (1 / m) * self.phi.T @ Ycenter.T
        if self.phi_adj_phi is None:
            self.phi_adj_phi = (1 / m) * self.phi.T @ self.phi
        n = len(X)
        d = len(self.B)
        alpha = sb04qd(n, d, K.numpy() / (self.regu * n), self.B.numpy() @ self.phi_adj_phi.numpy(), Yproj.T / (self.regu * n))
        self.alpha = torch.from_numpy(alpha.T)
    
    def predict_coefs(self, X, K=None):
        """
        Parameters
        ----------
        X : array_like
            Input data, of shape [n_samples, n_features]

        Returns
        -------
        torch.Tensor
            Predicted coefficients, of shape [n_atoms, n_samples]
        """
        if K is None:
            K = self.kernel(self.X, X)
        return self.B @ self.alpha @ K

    def predict(self, X, K=None):
        """
        Parameters
        ----------
        X : array_like
            Input data, of shape [n_samples, n_features]

        Returns
        -------
        torch.Tensor
            Predicted functions, of shape [n_samples, n_locations]
        """
        if K is None:
            K = self.kernel(self.X, X)
        if self.center_out:
            return (self.phi @ (self.B @ self.alpha @ K)).T + self.Ymean
        else:
            return (self.phi @ self.predict_coefs(X, K)).T

