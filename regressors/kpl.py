from tkinter import E
import numpy as np
from slycot import sb04qd
import torch

from optim import acc_proxgd
from optim.proxgd import acc_proxgd_restart


def center(Y, center_out):
    if center_out:
        if isinstance(Y, torch.Tensor):
            Ymean = Y.mean(dim=0).unsqueeze(0)
        else:
            Ymean = np.expand_dims(Y.mean(axis=0), axis=9)
        Ycenter = Y - Ymean
        return Ycenter, Ymean
    else:
        return Y, None


def fit_features(features, refit_features, X, K):
    if not features.fitted:
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
        self.center_out = center_out
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
        if isinstance(K, torch.Tensor):
            alpha = sb04qd(n, d, K.numpy() / (self.regu * n), self.B.numpy()
                           @ self.phi_adj_phi.numpy(), Yproj.T / (self.regu * n))
            self.alpha = torch.from_numpy(alpha.T)
        else:
            alpha = sb04qd(n, d, K / (self.regu * n), self.B @
                           self.phi_adj_phi, Yproj.T / (self.regu * n))
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
        self.center_out = center_out
        self.Ymean = None
        self.refit_features = refit_features

    def forget_phi(self):
        self.phi = None

    def set_phi(self, phi):
        self.phi = phi
        self.phi_adj_phi = (1 / self.phi.shape[0]) * self.phi.T @ self.phi

    def set_features(self, features):
        self.features = features

    def fit(self, X, Y, K=None, refit_features=False):
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
        fit_features(
            self.features, self.refit_features or refit_features, X, K)
        Z = self.features(X, K)
        q = Z.shape[1]
        # Make sure we have a Gram matrix
        if self.phi_adj_phi is None:
            self.phi_adj_phi = (1 / m) * self.phi.T @ self.phi
        if isinstance(K, torch.Tensor):
            alpha = sb04qd(q, d, (Z.T @ Z).numpy() / (self.regu * n),
                           self.phi_adj_phi.numpy(), Z.T.numpy() @ Yproj.T.numpy() / (self.regu * n))
            self.alpha = torch.from_numpy(alpha.T)
        else:
            alpha = sb04qd(q, d, Z.T @ Z / (self.regu * n),
                           self.phi_adj_phi, Z.T @ Yproj / (self.regu * n))
            self.alpha = alpha.T

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


# class FeaturesKPLDictsel:

#     def __init__(self, regu, features, phi, phi_adj_phi=None, center_out=False, refit_features=False):

#         super().__init__()
#         self.features = features
#         self.regu = regu
#         self.alpha = None
#         self.phi = phi
#         self.phi_adj_phi = phi_adj_phi
#         self.center_out= center_out
#         self.Ymean = None
#         self.refit_features = refit_features
#         self.n = None

#     def forget_phi(self):
#         self.phi = None

#     def set_phi(self, phi):
#         self.phi = phi

#     def grad(self, alpha):
#         val = (2 / self.n) * (self.phi_adj_phi @ alpha @ self.ZTZ - self.Yproj @ self.Z)
#         if torch.isinf((val ** 2).sum()):
#             raise ValueError("Gradient with infinite norm")
#         return val

#     def prox(self, alpha, stepsize):
#         norms = torch.norm(alpha, p=2, dim=1)
#         # print(norms)
#         thresh = torch.maximum(1 - (stepsize * self.regu) / norms, torch.tensor(0)).unsqueeze(1)
#         val = alpha * thresh
#         return val

#     # def prox(self, alpha, stepsize):
#     #     return (1 / (1 + self.regu * stepsize)) * alpha

#     def obj(self, alpha):
#         val = (1 / self.n) * (((self.phi @ (alpha @ self.Z.T)).T - self.Y) ** 2).sum()
#         if torch.isinf(val):
#             raise ValueError("Objective is infinite")
#         return val

#     def full_obj(self, alpha):
#         val = self.obj(alpha) + self.regu * torch.sqrt((alpha ** 2).sum(dim=1)).sum()
#         return val

#     def fit(self, X, Y, K=None, alpha0=None, n_epoch=20000, tol=1e-4, beta=0.5, acc_temper=20, monitor=None, stepsize0=0.1):
#         """
#         Parameters
#         ----------
#         X : array_like
#             Input data, of shape [n_samples, n_features]
#         Y : array_like
#             Output functions, of shape [n_samples, n_locations]
#         """
#         n = len(X)
#         m = Y.shape[1]
#         self.n = n
#         # Center if relevant
#         Ycenter, self.Ymean = center(Y, self.center_out)
#         # Project on output data and memorize needed quantities
#         self.Yproj = (1 / m) * self.phi.T @ Y.T
#         self.Y = Y
#         # Make sure features are fit and memorize needed quantities
#         fit_features(self.features, self.refit_features, X, K)
#         self.Z = self.features(X, K)
#         self.ZTZ = self.Z.T @ self.Z
#         # Make sure we have a Gram matrix and memorize it
#         if self.phi_adj_phi is None:
#             self.phi_adj_phi = (1 / m) * self.phi.T @ self.phi
#         if alpha0 is None:
#             alpha0 = torch.normal(0, 1, (self.phi.shape[1], self.features.n_features))
#         alpha, monitored = acc_proxgd_restart(
#             alpha0, self.prox, self.obj, self.full_obj, self.grad, n_epoch=n_epoch, tol=tol, beta=beta, acc_temper=acc_temper, monitor=monitor, stepsize0=stepsize0)
#         self.alpha = alpha
#         return monitored

#     def predict_coefs(self, X, K=None):
#         """
#         Parameters
#         ----------
#         X : array_like
#             Input data, of shape [n_samples, n_features]

#         Returns
#         -------
#         torch.Tensor
#             Predicted coefficients, of shape [n_atoms, n_samples]
#         """
#         Z = self.features(X, K)
#         return self.alpha @ Z.T

#     def predict(self, X, K=None):
#         """
#         Parameters
#         ----------
#         X : array_like
#             Input data, of shape [n_samples, n_features]

#         Returns
#         -------
#         torch.Tensor
#             Predicted functions, of shape [n_samples, n_locations]
#         """
#         if self.center_out:
#             return (self.phi @ self.predict_coefs(X, K)).T + self.Ymean
#         else:
#             return (self.phi @ self.predict_coefs(X, K)).T


class FeaturesKPLDictsel:

    def __init__(self, regu, regu_dictsel, features, phi, phi_adj_phi=None, center_out=False, refit_features=False):
        self.features = features
        self.regu = regu
        self.regu_dictsel = regu_dictsel
        self.alpha = None
        self.phi = phi
        self.phi_adj_phi = phi_adj_phi
        self.center_out = center_out
        self.Ymean = None
        self.refit_features = refit_features
        self.n = None

    def forget_phi(self):
        self.phi = None

    def set_phi(self, phi):
        self.phi = phi

    def set_features(self, features):
        self.features = features

    def grad(self, alpha):
        val = (2 / self.n) * (self.phi_adj_phi @ alpha @ self.ZTZ -
                              self.Yproj @ self.Z) + 2 * self.regu * self.phi_adj_phi @ alpha
        return val

    def prox(self, alpha, stepsize):
        if isinstance(alpha, torch.Tensor):
            norms = torch.norm(alpha, p=2, dim=1)
            thresh = torch.maximum(
                1 - (stepsize * self.regu_dictsel) / norms, torch.tensor(0)).unsqueeze(1)
        else:
            norms = np.linalg.norm(alpha, ord=2, axis=1)
            thresh = np.expand_dims(np.maximum(
                1 - (stepsize * self.regu_dictsel) / norms, 0), axis=1)
        val = alpha * thresh
        return val

    def obj(self, alpha):
        if isinstance(alpha, torch.Tensor):
            val = (1 / self.n) * (((self.phi @ (alpha @ self.Z.T)).T - self.Y) **
                                  2).sum() + self.regu * torch.diag(alpha.T @ self.phi_adj_phi @ alpha).sum()
        else:
            val = (1 / self.n) * (((self.phi @ (alpha @ self.Z.T)).T - self.Y) **
                                  2).sum() + self.regu * np.diag(alpha.T @ self.phi_adj_phi @ alpha).sum()
        return val

    def full_obj(self, alpha):
        if isinstance(alpha, torch.Tensor):
            val = self.obj(alpha) + self.regu_dictsel * \
                torch.sqrt((alpha ** 2).sum(dim=1)).sum()
        else:
            val = self.obj(alpha) + self.regu_dictsel * \
                np.sqrt((alpha ** 2).sum(axis=1)).sum()
        return val

    def fit(self, X, Y, K=None, alpha0=None, n_epoch=20000, tol=1e-4, beta=0.5, acc_temper=20, monitor=None, stepsize0=0.1, refit_features=False):
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
        fit_features(
            self.features, self.refit_features or refit_features, X, K)
        self.Z = self.features(X, K)
        self.ZTZ = self.Z.T @ self.Z
        # Make sure we have a Gram matrix and memorize it
        if self.phi_adj_phi is None:
            self.phi_adj_phi = (1 / m) * self.phi.T @ self.phi
        if alpha0 is None:
            if isinstance(X, torch.Tensor):
                alpha0 = torch.normal(
                    0, 1, (self.phi.shape[1], self.features.n_features))
            else:
                alpha0 = np.random.normal(
                    0, 1, (self.phi.shape[1], self.features.n_features))
        alpha, monitored = acc_proxgd_restart(
            alpha0, self.prox, self.obj, self.full_obj, self.grad, n_epoch=n_epoch, tol=tol, beta=beta, acc_temper=acc_temper, monitor=monitor, stepsize0=stepsize0)
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


class FeaturesKPLWorking:

    def __init__(self, regu, regu_dictsel, features, phi, phi_adj_phi=None, center_out=False, refit_features=False, regu_init=1e-8):
        self.features = features
        self.regu = regu
        self.regu_dictsel = regu_dictsel
        self.regu_init = regu_init
        self.alpha = None
        self.phi = phi
        self.phi_adj_phi = phi_adj_phi
        self.center_out = center_out
        self.Ymean = None
        self.refit_features = refit_features
        self.n = None
        self.working = None

    def forget_phi(self):
        self.phi = None

    def set_phi(self, phi):
        self.phi = phi
        self.phi_adj_phi = (1 / self.phi.shape[0]) * self.phi.T @ self.phi

    def grad(self, alpha, working=True):
        if working:
            sub_gram = self.phi_adj_phi[self.working][:, self.working]
            sub_Yproj = self.Yproj[self.working]
        else:
            sub_gram = self.phi_adj_phi
            sub_Yproj = self.Yproj
        val = (2 / self.n) * (sub_gram @ alpha @ self.ZTZ -
                              sub_Yproj @ self.Z) + 2 * self.regu * sub_gram @ alpha
        return val

    def prox(self, alpha, stepsize):
        if isinstance(alpha, torch.Tensor):
            norms = torch.norm(alpha, p=2, dim=1)
            thresh = torch.maximum(
                1 - (stepsize * self.regu_dictsel) / norms, torch.tensor(0)).unsqueeze(1)
        else:
            norms = np.linalg.norm(alpha, ord=2, axis=1)
            thresh = np.expand_dims(np.maximum(
                1 - (stepsize * self.regu_dictsel) / norms, 0), axis=1)
        val = alpha * thresh
        return val

    # def prox(self, alpha, stepsize):
    #     return (1 / (1 + self.regu * stepsize)) * alpha

    def obj(self, alpha):
        sub_gram = self.phi_adj_phi[self.working][:, self.working]
        sub_phi = self.phi[:, self.working]
        if isinstance(alpha, torch.Tensor):
            val = (1 / self.n) * (((sub_phi @ (alpha @ self.Z.T)).T - self.Y) **
                                  2).sum() + self.regu * torch.diag(alpha.T @ sub_gram @ alpha).sum()
        else:
            val = (1 / self.n) * (((sub_phi @ (alpha @ self.Z.T)).T - self.Y) **
                                  2).sum() + self.regu * np.diag(alpha.T @ sub_gram @ alpha).sum()
        return val

    def full_obj(self, alpha):
        if isinstance(alpha, torch.Tensor):
            val = self.obj(alpha) + self.regu_dictsel * \
                torch.sqrt((alpha ** 2).sum(dim=1)).sum()
        else:
            val = self.obj(alpha) + self.regu_dictsel * \
                np.sqrt((alpha ** 2).sum(axis=1)).sum()
        return val

    def fit(self, X, Y, K=None, alpha0=None, n_epoch=20000, tol=1e-4, beta=0.5, acc_temper=20, monitor=None, stepsize0=0.1, verbose=False):
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
        # Selection of first atom with maximum average absolute correlation with outputs
        if isinstance(X, torch.Tensor):
            corr = self.Yproj.abs().mean(dim=1)
            self.working = [torch.argmax(corr).item()]
        else:
            corr = np.abs(self.Yproj).mean(axis=1)
            self.working = [np.argmax(corr)]
        stop = False
        q = self.ZTZ.shape[0]
        d = self.phi_adj_phi.shape[0]
        while not stop:
            # Init alpha with closed-form
            sub_gram = self.phi_adj_phi[self.working][:, self.working]
            sub_Yproj = self.Yproj[self.working]
            dsub = len(self.working)
            if isinstance(X, torch.Tensor):
                alpha0 = sb04qd(q, dsub, self.ZTZ.numpy() / (self.regu_init * n), sub_gram.numpy(
                ), self.Z.T.numpy() @ sub_Yproj.T.numpy() / (self.regu_init * n))
                alpha0 = torch.Tensor(alpha0.T)
            else:
                alpha0 = sb04qd(q, dsub, self.ZTZ / (self.regu_init * n),
                                sub_gram, self.Z.T @ sub_Yproj.T / (self.regu_init * n))
                alpha0 = alpha0.T
            # Fit model on working set
            alpha, _ = acc_proxgd_restart(
                alpha0, self.prox, self.obj, self.full_obj, self.grad, n_epoch=n_epoch, tol=tol,
                beta=beta, acc_temper=acc_temper, monitor=monitor, stepsize0=stepsize0, verbose=verbose)
            # alpha, _ = acc_proxgd(
            #     alpha0, self.prox, self.obj, self.full_obj, self.grad, n_epoch=n_epoch, tol=tol,
            #     beta=beta, acc_temper=acc_temper, monitor=monitor, stepsize0=stepsize0)
            # Check global optimality
            if isinstance(X, torch.Tensor):
                alpha_full = torch.zeros((d, q))
                alpha_full[self.working] = alpha
                grad = self.grad(alpha_full, working=False)
                norms = torch.norm(grad, dim=1)
                maxnorm = torch.max(norms)
            else:
                alpha_full = np.zeros((d, q))
                alpha_full[self.working] = alpha
                grad = self.grad(alpha_full, working=False)
                norms = np.linalg.norm(grad, axis=1)
                maxnorm = np.max(norms)
            # Dual norm is (infinity, 2)-norm, global optimality iff <= lambda
            if maxnorm <= self.regu2:
                stop = True
                self.alpha = alpha_full
            # If not global optimal, add atom which violates above optimality condition most
            else:
                if isinstance(X, torch.Tensor):
                    to_add = torch.argmax(norms).item()
                else:
                    to_add = np.argmax(norms)
                # Avoid infinite loop if regularization is too low for global optimality condition to be satfisfied
                if to_add in self.working:
                    stop = True
                    self.alpha = alpha_full
                self.working.append(to_add)

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
