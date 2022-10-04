import numpy as np
from slycot import sb04qd
import torch

from optim import acc_proxgd


def center(Y, center_out):
    if center_out:
        Ymean = Y.mean(dim=0).unsqueeze(0)
        Ycenter = Y - Ymean
        return Ycenter, Ymean
    else:
        return Y, None


def fit_features(features, refit_features, X, K):
    if not features.fit:
        features.fit(X, K)
    else:
        if refit_features:
            features.fit(X, K)


class SeparableKPL:
    """
    Parameters
    ----------
    kernel : callableq
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
        # Center output funtions if relevant
        Ycenter, self.Ymean = center(Y, self.center_out)
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
            return (self.phi @ self.predict_coefs(X, K)).T + self.Ymean
        else:
            return (self.phi @ self.predict_coefs(X, K)).T



class FeaturesKPL: 

    """
    Parameters
    ----------
    phi : torch.Tensor
        Discretized projection operator, of shape [n_locations, n_atoms]
    regu : float
        Regularization parameter
    center_out : bool
        Should output functions be centered
    """
    def __init__(self, regu, features, phi, phi_adj_phi=None, center_out=False, refit_features=False):

        super().__init__()
        self.features = features
        self.regu = regu
        self.alpha = None
        self.phi = phi
        self.phi_adj_phi = phi_adj_phi
        self.center_out= center_out
        self.Ymean = None
        self.refit_features = refit_features

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
        n = len(X)
        m = Y.shape[1]
        d = self.phi.shape[1]
        # Center if relevant
        Ycenter, self.Ymean = center(Y, self.center_out)
        # Project on output data
        Yproj = (1 / m) * self.phi.T @ Ycenter.T
        # Make sure features are fit
        fit_features(self.features, self.refit_features, X, K)
        q = self.features.n_features
        Z = self.features(X, K)
        # Make sure we have a Gram matrix
        if self.phi_adj_phi is None:
            self.phi_adj_phi = (1 / m) * self.phi.T @ self.phi
        alpha = sb04qd(q, d, (Z.T @ Z).numpy() / (self.regu * n), self.phi_adj_phi.numpy(), Z.T.numpy() @ Yproj.T.numpy() / (self.regu * n))
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
        Z = self.features(X, K)
        return self.alpha @ Z.T

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
        if self.center_out:
            return (self.phi @ self.predict_coefs(X, K)).T + self.Ymean
        else:
            return (self.phi @ self.predict_coefs(X, K)).T



class FeaturesKPLDictsel: 

    def __init__(self, regu, features, phi, phi_adj_phi=None, center_out=False, refit_features=False):

        super().__init__()
        self.features = features
        self.regu = regu
        self.alpha = None
        self.phi = phi
        self.phi_adj_phi = phi_adj_phi
        self.center_out= center_out
        self.Ymean = None
        self.refit_features = refit_features
        self.n = None

    def forget_phi(self):
        self.phi = None

    def set_phi(self, phi):
        self.phi = phi
    
    def grad(self, alpha):
        return (2 / self.n) * (self.phi_adj_phi @ alpha @ self.ZTZ - self.Yproj @ self.Z)
    
    def prox(self, alpha, stepsize):
        norms = torch.norm(alpha, p=2, dim=1)
        thresh = torch.maximum(1 - (stepsize * self.regu) / norms, torch.tensor(0)).unsqueeze(1)
        return alpha * thresh
    
    def obj(self, alpha):
        return (1 / self.n) * (((self.phi @ (alpha @ self.Z.T)).T - self.Y) ** 2).sum()

    def fit(self, X, Y, K=None, alpha0=None, n_epoch=20000, tol=1e-4, beta=0.5, d=20, monitor=None):
        """
        Parameters
        ----------
        X : array_like
            Input data, of shape [n_samples, n_features]
        Y : array_like
            Output functions, of shape [n_samples, n_locations]
        """
        n = len(X)
        m = Y.shape[1]
        self.n = n
        # Center if relevant
        Ycenter, self.Ymean = center(Y, self.center_out)
        # Project on output data and memorize needed quantities
        self.Yproj = (1 / m) * self.phi.T @ Y.T
        self.Y = Y
        # Make sure features are fit and memorize needed quantities
        fit_features(self.features, self.refit_features, X, K)
        self.Z = self.features(X, K)
        self.ZTZ = self.Z.T @ self.Z
        # Make sure we have a Gram matrix and memorize it
        if self.phi_adj_phi is None:
            self.phi_adj_phi = (1 / m) * self.phi.T @ self.phi
        if alpha0 is None:
            alpha0 = torch.zeros((self.phi.shape[1], self.features.n_features))
        alpha, monitored = acc_proxgd(alpha0, self.prox, self.obj, self.grad, n_epoch=n_epoch, tol=tol, beta=beta, d=d, monitor=monitor)
        self.alpha = alpha
        return monitored
    
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
        Z = self.features(X, K)
        return self.alpha @ Z.T

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
        if self.center_out:
            return (self.phi @ self.predict_coefs(X, K)).T + self.Ymean
        else:
            return (self.phi @ self.predict_coefs(X, K)).T