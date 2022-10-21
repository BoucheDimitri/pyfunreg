import numpy as np
from scipy import linalg
import torch


class NystromFeatures: 

    def __init__(self, kernel, n_features, seed, thresh=1e-6):
        self.kernel = kernel
        self.seed = seed
        self.n_features = n_features
        self.n_features_eff = None
        self.thresh = thresh
        self.eigvals, self.eigvecs = None, None
        self.inds = None
        self.Xnys = None
        self.fitted = False
        self.backend = "torch"

    def fit(self, X, K=None):
        if K is None:
            K = self.kernel(X, X)
        n = len(X)
        np.random.seed(self.seed)
        if self.n_features > n:
            self.n_features = n
        self.inds = np.random.choice(np.arange(0, n), self.n_features, replace=False)
        self.Xnys = X[self.inds]
        Knys = K[self.inds][:, self.inds]
        if isinstance(K, torch.Tensor):
            Knys = Knys.numpy()
        u, V = linalg.eigh(Knys)
        thresh_inds = np.argwhere(u > self.thresh).flatten()
        self.n_features_eff = len(thresh_inds)
        self.eigvals, self.eigvecs = u[thresh_inds], V[:, thresh_inds]
        self.fitted = True

    def __call__(self, X, K=None):
        if isinstance(X, torch.Tensor):
            if K is None:
                Ksub = self.kernel(self.Xnys, X).numpy()
            else:
                Ksub = K[self.inds].numpy()
            return torch.from_numpy(((np.diag(1 / np.sqrt(self.eigvals)) @ self.eigvecs.T) @ Ksub).T)
        else:
            if K is None:
                Ksub = self.kernel(self.Xnys, X)
            else:
                Ksub = K[self.inds]
            return ((np.diag(1 / np.sqrt(self.eigvals)) @ self.eigvecs.T) @ Ksub).T


class RandomFourierFeatures:

    def __init__(self, gamma, n_features, seed):
        self.gamma = gamma
        self.n_features = n_features
        self.input_dim = None
        self.seed = seed
        self.w, self.b = None, None
        self.fit = False
    
    def fit(self, X, K=None):
        self.input_dim = X.shape[1]
        np.random.seed(self.seed)
        self.w = torch.from_numpy(np.random.normal(0, 1, (self.input_dim, self.n_features)))
        self.b = torch.from_numpy(np.random.uniform(0, 2 * np.pi, (1, self.n_features)))
        self.fit = True
    
    def __call__(self, X, K=None):
        return np.sqrt(2 / self.n_features) * torch.cos(np.sqrt(2 * self.gamma) * X @ self.w + self.b)

