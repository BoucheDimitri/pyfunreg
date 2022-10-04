import numpy as np
from slycot import sb04qd
import torch



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



