from abc import ABC, abstractmethod
import torch
import numpy as np
from slycot import sb04qd
from optim.proxgd import acc_proxgd_restart


class DualFOR(ABC):

    def __init__(self, regu, kernel):
        self.regu = regu
        self.kernel = kernel
        self.losses = None
        self.n = None
        self.m = None
        self.Kx = None
        self.Ktheta = None
        self.Y = None
        self.alpha = None

    @abstractmethod
    def primal_obj(self, alpha):
        pass

    @abstractmethod
    def dual_obj_diff(self, alpha):
        pass

    # @abstractmethod
    # def dual_loss_full(self, alpha):
    #     pass

    @abstractmethod
    def dual_grad(self, alpha):
        pass

    @abstractmethod
    def prox_step(self, alpha, gamma=None):
        pass

    def sylvester_init(self, X, Y, thetas, Kx=None, Ktheta=None):
        if Kx is None:
            Kx = self.Kx
        if Ktheta is None:
            Ktheta = self.Ktheta
        if isinstance(X, torch.Tensor):
            alpha = sb04qd(self.n, self.m, 
                           Kx.numpy() / (self.lbda * self.n * self.m), 
                           Ktheta.numpy(), Y.numpy() / (self.lbda * self.model.n * self.model.m))
            self.alpha = torch.from_numpy(alpha)
        else:
            alpha = sb04qd(self.n, self.m, 
                           self.model.Kx / (self.lbda * self.model.n * self.model.m), 
                           self.model.Ktheta, Y / (self.lbda * self.model.n * self.model.m))
            self.alpha = alpha



class FORSpl(DualFOR):

    def __init__(self, regu, kernel):
        super().__init__(regu, kernel)

    def primal_obj(self, alpha):
        n = self.Kx.shape[0]
        m = self.Ktheta.shape[0]
        pred = (1 / (self.regu * n * m)) * self.Kx @ alpha @ self.Ktheta
        return ((pred - self.Y) ** 2).mean()

    def dual_obj_diff(self, alpha):
        A = 0.5 * alpha @ alpha.T
        B = - alpha @ self.Y.T
        cste = 0.5 / (self.regu * self.n * self.m)
        C = cste * self.Kx @ alpha @ self.Ktheta @ alpha.T
        if isinstance(alpha, torch.Tensor):
            return torch.trace(A + B + C)
        else:
            return np.trace(A + B + C)

    # def dual_loss_full(self, alpha):
    #     return self.dual_loss_diff(alpha)

    def dual_grad(self, alpha):
        A = alpha
        B = - self.Y
        cste = 1 / (self.regu * self.n * self.m)
        C = cste * self.Kx @ alpha @ self.Ktheta
        return A + B + C

    def prox_step(self, alpha, gamma=None):
        return alpha

    def fit_sylvester(self, X, Y, thetas, Kx=None, Ktheta=None):
        if Kx is None:
            Kx = self.Kx
        if Ktheta is None:
            Ktheta = self.Ktheta

        if isinstance(X, torch.Tensor):
            alpha = sb04qd(self.n, self.m, 
                           Kx.numpy() / (self.lbda * self.n * self.m), 
                           Ktheta.numpy(), Y.numpy() / (self.lbda * self.model.n * self.model.m))
            self.alpha = torch.from_numpy(alpha)
        else:
            alpha = sb04qd(self.n, self.m, 
                           self.model.Kx / (self.lbda * self.model.n * self.model.m), 
                           self.model.Ktheta, Y / (self.lbda * self.model.n * self.model.m))
            self.alpha = alpha


class FeaturesKPLOtherLoss:

    def __init__(self, regu, loss, features, phi, optimizer, center_out=False, refit_features=False, sylvester_init=True):
        self.features = features
        self.regu = regu
        self.alpha = None
        self.phi = phi
        self.center_out = center_out
        self.Ymean = None
        self.refit_features = refit_features
        self.n = None
        self.loss = loss
        self.optimizer = optimizer
        self.sylvester_init = sylvester_init

    def forget_phi(self):
        self.phi = None

    def set_phi(self, phi):
        self.phi = phi
        self.phi_adj_phi = (1 / self.phi.shape[0]) * self.phi.T @ self.phi

    def set_features(self, features):
        self.features = features

    def set_loss(self, loss):
        self.loss = loss

    def grad(self, alpha):
        G = self.loss.grad(self.Z @ alpha.T @ self.phi.T -
                           self.Y) @ (self.phi / self.phi.shape[0])
        return G.T @ self.Z + 2 * self.regu * alpha

    def obj(self, alpha):
        return self.loss(self.Z @ alpha.T @ self.phi.T - self.Y) + self.regu * (alpha ** 2).sum()

    def full_obj(self, alpha):
        return self.obj(alpha)

    def prox(self, alpha, gamma):
        return alpha

    def fit(self, X, Y, K=None, alpha0=None, refit_features=False):
        n = len(X)
        m = Y.shape[1]
        self.n = n
        # Center if relevant
        Ycenter, self.Ymean = center(Y, self.center_out)
        self.Y = Y
        # Make sure features are fit and memorize needed quantities
        fit_features(
            self.features, self.refit_features or refit_features, X, K)
        self.Z = self.features(X, K)
        if self.sylvester_init and alpha0 is None:
            phi_adj_phi = (1 / m) * self.phi.T @ self.phi
            d = self.phi.shape[1]
            q = self.Z.shape[1]
            Yproj = (1 / m) * self.phi.T @ Ycenter.T
            if isinstance(X, torch.Tensor):
                alpha0 = sb04qd(q, d, (self.Z.T @ self.Z).numpy() / (self.regu * n),
                                phi_adj_phi.numpy(), self.Z.T.numpy() @ Yproj.T.numpy() / (self.regu * n))
                alpha0 = torch.from_numpy(alpha0.T)
            else:
                alpha0 = sb04qd(q, d, self.Z.T @ self.Z / (self.regu * n), phi_adj_phi, self.Z.T @ Yproj.T / (self.regu * n))
                alpha0 = alpha0.T
        if alpha0 is None:
            if isinstance(X, torch.Tensor):
                alpha0 = torch.normal(
                    0, 1, (self.phi.shape[1], self.features.n_features_eff))
            else:
                alpha0 = np.random.normal(
                    0, 1, (self.phi.shape[1], self.features.n_features_eff))
        alpha, monitored = self.optimizer(
            alpha0, self.prox, self.obj, self.full_obj, self.grad)
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
