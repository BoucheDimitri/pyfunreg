import numpy as np
from abc import ABC, abstractmethod
import torch

from kernel import RandomFourierFeatures


class Basis(ABC):
    """
    Abstract class for set of basis functions

    Parameters
    ----------
    n_basis: int
        Number of basis functions
    domain: array-like, shape = [2,]
        Bounds of the interval of definition
    """
    def __init__(self, n_basis, domain):
        self.n_basis = n_basis
        self.gram_matrix = None
        self.domain = np.array(domain)
        super().__init__()

    @abstractmethod
    def compute_Phi(self, thetas):
        """
        Evaluate the set of basis functions on a given set of values

        Parameters
        ----------
        thetas : array-like, shape = [n_locations, ]
            Locations of evaluation of the functions

        Returns
        -------
        array-like, shape=[n_locations, n_basis]
            Matrix of evaluations at the locations thetas for the basis functions
        """
        pass

    @abstractmethod
    def get_Phi_adj_Phi(self):
        pass


class FourierBasis(Basis):
    """
    Fourier basis

    Parameters
    ----------
    lower_freq: int
        Minimum frequency to consider
    upper_freq: int
        Maximum frequency to consider
    domain: array-like, shape = [input_dim, 2]
        Bounds for the domain of the basis function
    """

    def __init__(self, lower_freq, upper_freq, domain):
        self.add_constant = lower_freq == 0
        self.cos_freqs = np.array(np.arange(lower_freq + int(self.add_constant), upper_freq))
        self.sin_freqs = np.array(np.arange(lower_freq + int(self.add_constant), upper_freq))
        n_basis = len(self.cos_freqs) + len(self.sin_freqs) + int(self.add_constant)
        super().__init__(n_basis, domain)

    @staticmethod
    def cos_atom(n, a, b, theta):
        return (1 / np.sqrt((b - a) / 2)) * np.cos((2 * np.pi * n * (theta - a)) / (b - a))

    @staticmethod
    def sin_atom(n, a, b, theta):
        return (1 / np.sqrt((b - a) / 2)) * np.sin((2 * np.pi * n * (theta - a)) / (b - a))

    @staticmethod
    def constant_atom(a, b, theta):
        return 1 / np.sqrt(b - a) * np.ones(theta.shape)

    def compute_Phi(self, thetas):
        mat = np.zeros((thetas.shape[0], self.n_basis))
        a, b = self.domain[0], self.domain[1]
        count = 0
        if self.add_constant:
            mat[:, 0] = np.array([FourierBasis.constant_atom(a, b, thetas)])
            count += 1
        for freq in self.cos_freqs:
            mat[:, count] = np.array([FourierBasis.cos_atom(freq, a, b, thetas)])
            count += 1
        for freq in self.sin_freqs:
            mat[:, count] = np.array([FourierBasis.sin_atom(freq, a, b, thetas)])
            count += 1
        return mat

    def get_Phi_adj_Phi(self):
        return np.eye(self.n_basis)


class RandomFourierBasis(Basis):

    def __init__(self, gamma, seed, n_basis, domain):
        self.rffs = RandomFourierFeatures(gamma, n_basis, seed)
        self.rffs.fit(np.zeros((1, 1)))
        super().__init__(n_basis, domain)
    
    def compute_Phi(self, thetas):
        return self.rffs(torch.from_numpy(thetas).unsqueeze(1)).numpy()
    
    def compute_Phi_adj_Phi(self, thetas):
        Phi = self.compute_Phi(thetas)
        self.gram_matrix = (1 / len(thetas)) * Phi.T @ Phi

    def get_Phi_adj_Phi(self):
        return self.gram_matrix










# from abc import ABC, abstractmethod
# import numpy as np
# import functools
# from scipy.interpolate import BSpline
# from sklearn.decomposition import DictionaryLearning
# import pywt

# from functional_data import smoothing
# from functional_data import fpca
# from functional_data import functional_algebra as falgebra
# from functional_data import discrete_functional_data as disc_fd


# def constant_function(domain):
#     norm_const = 1 / (domain[0, 1] - domain[0, 0])

#     def const_func(z):
#         if np.ndim(z) == 0:
#             return norm_const
#         else:
#             return norm_const * np.ones(z.shape[0])
#     return const_func


# # ######################## Abstract classes ############################################################################

# class Basis(ABC):
#     """
#     Abstract class for set of basis functions

#     Parameters
#     ----------
#     n_basis: int
#         Number of basis functions
#     domain: array-like, shape = [2,]
#         Bounds of the interval of definition

#     Attributes
#     ----------
#     n_basis: int
#         Number of basis functions
#     domain: array-like, shape = [2,]
#         Bounds of the interval of definition
#     """
#     def __init__(self, n_basis, domain):
#         self.n_basis = n_basis
#         self.gram_matrix = None
#         self.domain = np.array(domain)
#         self._gram_mat = None
#         super().__init__()

#     @abstractmethod
#     def compute_matrix(self, X):
#         """
#         Evaluate the set of basis functions on a given set of values

#         Parameters
#         ----------
#         X : array-like, shape = [n_input, input_dim]
#             The input data

#         Returns
#         -------
#         array-like, shape=[n_input, n_basis]
#             Matrix of evaluations of the inputs for all basis-function
#         """
#         pass

#     @abstractmethod
#     def get_gram_matrix(self):
#         pass


# class DataDependantBasis(Basis):
#     """
#     Abstract class for set of basis functions

#     Parameters
#     ----------
#     n_basis: int
#         Number of basis functions
#     domain: array-like, shape = [2,]
#         Bounds of the interval of definition

#     Attributes
#     ----------
#     n_basis: int
#         Number of basis functions
#     domain: array-like, shape = [2,]
#         Bounds of the interval of definition
#     """
#     def __init__(self, n_basis, domain):
#         super().__init__(n_basis, domain)

#     @abstractmethod
#     def fit(self, Ylocs, Yobs):
#         """
#         Fit the basis to the data

#         Parameters
#         ----------
#         Ylocs: iterable of array-like
#             The input locations, len = n_samples and for the i-th sample, Ylocs[i] has shape = [n_observations_i, 1]
#         Yobs: iterable of array-like
#             The observations len = n_samples and for the i-th sample, Yobs[i] has shape = [n_observations_i, ]
#         """
#         pass


# # ######################## Classic Bases ###############################################################################


# class RandomFourierFeatures(Basis):
#     """
#     Random Fourier Features basis

#     Parameters
#     ----------
#     n_basis: int
#         Number of basis functions
#     domain: array-like, shape = [2,]
#         Bounds of the interval of definition
#     bandwidth: float
#         Bandwidth parameter of approximated kernel
#     seed: int
#         Seed to initialize for each random draw

#     Attributes
#     ----------
#     n_basis: int
#         Number of basis functions
#     domain: array-like, shape = [2,]
#         Bounds of the interval of definition
#     bandwidth: float
#         Bandwidth parameter of approximated kernel
#     seed: int
#         Seed to initialize for each random draw (for reproducible experiments)
#     w: array-like
#         Random weights
#     b: array-like
#         Random intercept
#     """
#     def __init__(self, n_basis, domain, bandwidth, seed=0,
#                  compute_gram=True, normalize=True, add_constant=True):
#         super().__init__(n_basis + int(add_constant), domain)
#         self.add_constant = add_constant
#         self.bandwidth = bandwidth
#         self.seed = seed
#         np.random.seed(self.seed)
#         self.w = np.random.normal(0, 1, (self.input_dim, self.n_basis - int(add_constant)))
#         np.random.seed(seed)
#         self.b = np.random.uniform(0, 2 * np.pi, (1, self.n_basis - int(add_constant)))
#         if add_constant:
#             self.w = np.concatenate((np.zeros((self.input_dim, 1)), self.w), axis=1)
#             self.b = np.concatenate((np.zeros((1, 1)), self.b), axis=1)
#         self.normalize = normalize
#         self.gram_mat = None
#         self.norms = None
#         self.compute_gram = compute_gram
#         if normalize:
#             self.compute_norms()
#         if compute_gram:
#             self.compute_gram_matrix()
#         # self.compute_gram_matrix_bis()

#     def get_atoms_configs(self, inds):
#         return [(self.w[:, ind], self.b[:, ind]) for ind in inds]

#     def compute_norms(self, n_approx=500):
#         space = np.linspace(self.domain[0, 0], self.domain[0, 1], n_approx)
#         mat = self.compute_matrix_nonnormalized(space)
#         self.norms = np.sqrt(np.mean(mat ** 2, axis=0))

#     def update_norms(self, n_added, n_approx=500):
#         inds = np.arange(self.n_basis - n_added, self.n_basis)
#         space = np.linspace(self.domain[0, 0], self.domain[0, 1], n_approx)
#         mat = self.compute_matrix_nonnormalized(space, inds)
#         self.norms = np.concatenate((self.norms, np.sqrt(np.mean(mat ** 2, axis=0))))

#     def compute_gram_matrix(self, n_approx=500):
#         space = np.linspace(self.domain[0, 0], self.domain[0, 1], n_approx)
#         if self.normalize:
#             mat = self.compute_matrix(space)
#         else:
#             mat = self.compute_matrix_nonnormalized(space)
#         self.gram_mat = ((self.domain[0, 1] - self.domain[0, 0])/n_approx) * mat.T.dot(mat)

#     def compute_matrix_nonnormalized(self, X, inds=None):
#         n = X.shape[0]
#         if inds is None:
#             if X.ndim == 1:
#                 return np.sqrt(2 / self.n_basis) * np.cos(self.bandwidth * X.reshape((n, 1)).dot(self.w) + self.b)
#             else:
#                 return np.sqrt(2 / self.n_basis) * np.cos(self.bandwidth * X.dot(self.w) + self.b)
#         else:
#             if X.ndim == 1:
#                 return np.sqrt(2 / self.n_basis) * np.cos(self.bandwidth * X.reshape((n, 1)).dot(self.w[:, inds]) + self.b[:, inds])
#             else:
#                 return np.sqrt(2 / self.n_basis) * np.cos(self.bandwidth * X.dot(self.w[:, inds]) + self.b[:, inds])

#     def compute_matrix(self, X):
#         n = X.shape[0]
#         if self.normalize:
#             if X.ndim == 1:
#                 return (1 / np.expand_dims(self.norms, axis=0)) \
#                        * np.sqrt(2 / self.n_basis) * np.cos(self.bandwidth * X.reshape((n, 1)).dot(self.w) + self.b)
#             else:
#                 return (1 / np.expand_dims(self.norms, axis=0)) * \
#                        np.sqrt(2 / self.n_basis) * np.cos(self.bandwidth * X.dot(self.w) + self.b)
#         else:
#             if X.ndim == 1:
#                 return np.sqrt(2 / self.n_basis) * np.cos(self.bandwidth * X.reshape((n, 1)).dot(self.w) + self.b)
#             else:
#                 return np.sqrt(2 / self.n_basis) * np.cos(self.bandwidth * X.dot(self.w) + self.b)


#     def get_atom(self, i):
#         return lambda x: np.sqrt(2 / self.n_basis) * np.cos(self.bandwidth * self.w[:, i].dot(x) + self.b[:, i])

#     def get_gram_matrix(self):
#         return self.gram_mat


# class BasisFromSmoothFunctions(Basis):
#     """
#     Basis from list of vectorized functions

#     Parameters
#     ----------
#     funcs: iterable
#         Iterable containing callables that can be called on array-like inputs
#     input_dim: int
#         The input dimension for the functions
#     domain: array-like, shape = [2, ]
#         Bounds for the domain of the basis function

#     Attributes
#     ----------
#     n_basis: int
#         Number of basis functions
#     domain: array-like, shape = [2, ]
#         Bounds for the domain of the basis function
#     funcs: iterable
#         Iterable containing callables that can be called on array-like inputs
#     add_constant: bool
#         Should the constant function be automatically be added
#     """
#     def __init__(self, funcs, domain, add_constant=False):
#         n_basis = len(funcs)
#         self.funcs = funcs
#         self.add_constant = add_constant
#         super().__init__(n_basis + int(add_constant), domain)

#     def __getitem__(self, item):
#         return self.funcs[item]

#     def compute_matrix(self, X):
#         n = X.shape[0]
#         mat = np.zeros((n, self.n_basis))
#         for i in range(self.n_basis - int(self.add_constant)):
#             mat[:, i] = self.funcs[i](X)
#         if self.add_constant:
#             mat[:, -1] = 1
#         return mat


# class FourierBasis(Basis):
#     """
#     Abstract class for set of basis functions

#     Parameters
#     ----------
#     lower_freq: int
#         Minimum frequency to consider
#     upper_freq: int
#         Maximum frequency to consider
#     domain: array-like, shape = [input_dim, 2]
#         Bounds for the domain of the basis function

#     Attributes
#     ----------
#     n_basis: int
#         Number of basis functions
#     input_dim: int
#         The number of dimensions of the input space
#     domain: array-like, shape = [input_dim, 2]
#         Bounds for the domain of the basis function
#     freqs: tuple, len = 2
#         Frequencies included in the basis
#     """

#     def __init__(self, lower_freq, upper_freq, domain):
#         self.add_constant = lower_freq == 0
#         self.cos_freqs = np.array(np.arange(lower_freq + int(self.add_constant), upper_freq))
#         self.sin_freqs = np.array(np.arange(lower_freq + int(self.add_constant), upper_freq))
#         n_basis = len(self.cos_freqs) + len(self.sin_freqs) + int(self.add_constant)
#         super().__init__(n_basis, domain)

#     @staticmethod
#     def cos_atom(n, a, b, x):
#         return (1 / np.sqrt((b - a) / 2)) * np.cos((2 * np.pi * n * (x - a)) / (b - a))

#     @staticmethod
#     def sin_atom(n, a, b, x):
#         return (1 / np.sqrt((b - a) / 2)) * np.sin((2 * np.pi * n * (x - a)) / (b - a))

#     @staticmethod
#     def constant_atom(a, b, x):
#         return 1 / np.sqrt(b - a) * np.ones(x.shape)

#     def compute_matrix(self, X):
#         n = X.shape[0]
#         mat = np.zeros((n, self.n_basis))
#         a = self.domain[0, 0]
#         b = self.domain[0, 1]
#         count = 0
#         if self.add_constant:
#             mat[:, 0] = np.array([FourierBasis.constant_atom(a, b, np.squeeze(X))])
#             count += 1
#         for freq in self.cos_freqs:
#             mat[:, count] = np.array([FourierBasis.cos_atom(freq, a, b, np.squeeze(X))])
#             count += 1
#         for freq in self.sin_freqs:
#             mat[:, count] = np.array([FourierBasis.sin_atom(freq, a, b, np.squeeze(X))])
#             count += 1
#         return mat

#     def get_gram_matrix(self):
#         return np.eye(self.n_basis)


# class BSplineUniscaleBasis(Basis):
#     """
#     Parameters
#     ----------
#     domain: array-like, shape = [1, 2]
#         the domain of interest
#     n_basis: int
#         number of basis to consider (regular repartition into locs_bounds
#     locs_bounds: array-like, shape = [1, 2]
#         the bounds for the peaks locations of the splines
#     width: int
#         the width of the splines
#     bounds_disc: bool
#         should the knots outside of the domain be thresholded to account for possible out of domain discontinuity
#     order: int
#         the order of the spline, 3 is for cubic spline for instance.
#     """
#     def __init__(self, domain, n_basis, locs_bounds, width=1.0, bounds_disc=False,
#                  order=3, add_constant=True):
#         self.locs_bounds = locs_bounds
#         self.bounds_disc = bounds_disc
#         self.order = order
#         self.width = width
#         self.knots = BSplineUniscaleBasis.knots_generator(domain, n_basis, locs_bounds, width, bounds_disc, order)
#         self.splines = [falgebra.NoNanWrapper(BSpline.basis_element(self.knots[i], extrapolate=False))
#                         for i in range(len(self.knots))]
#         self.add_constant = add_constant
#         self.norms = [np.sqrt(integration.func_scalar_prod(sp, sp, domain)) for sp in self.splines]
#         # self.norms = [np.sqrt(integration.func_scalar_prod(sp, sp, domain)[0]) for sp in self.splines]
#         input_dim = 1
#         super().__init__(n_basis + int(self.add_constant), input_dim, domain)
#         self.gram_mat = self.gram_matrix()

#     @staticmethod
#     def knots_generator(domain, n_basis, locs_bounds, width=1, bounds_disc=False, order=3):
#         locs = np.linspace(locs_bounds[0], locs_bounds[1], n_basis, endpoint=True)
#         pace = width / (order + 1)
#         cardinal_knots = np.arange(-width / 2, width / 2 + pace, pace)
#         if not bounds_disc:
#             knots = [cardinal_knots + loc for loc in locs]
#         else:
#             knots = []
#             for loc in locs:
#                 knot = cardinal_knots + loc
#                 knot[knot < domain[0, 0]] = domain[0, 0]
#                 knot[knot > domain[0, 1]] = domain[0, 1]
#                 knots.append(knot)
#         return knots

#     def gram_matrix(self):
#         gram_mat = np.zeros((self.n_basis, self.n_basis))
#         funcs = self.splines.copy()
#         if self.add_constant:
#             funcs.append(constant_function(self.domain))
#         for i in range(self.n_basis):
#             for j in range(i, self.n_basis):
#                 esti_scalar = integration.func_scalar_prod(funcs[i], funcs[j], self.domain)
#                 gram_mat[i, j] = (1 / (self.norms[i] * self.norms[j])) * esti_scalar
#                 gram_mat[j, i] = (1 / (self.norms[i] * self.norms[j])) * esti_scalar
#                 # gram_mat[i, j] = esti_scalar[0]
#                 # gram_mat[j, i] = esti_scalar[0]
#         return gram_mat

#     def get_gram_matrix(self):
#         return self.gram_mat

#     def compute_matrix(self, X):
#         if X.ndim == 1:
#             Xreshaped = np.expand_dims(X, axis=1)
#         else:
#             Xreshaped = X
#         evals = [(1 / self.norms[i]) * self.splines[i](Xreshaped) for i in range(len(self.norms))]
#         # evals = [sp(Xreshaped) for sp in self.splines]
#         constant = np.ones((Xreshaped.shape[0], 1))
#         if self.add_constant:
#             evals.append(constant)
#         evals = np.concatenate(evals, axis=1)
#         evals[np.isnan(evals)] = 0
#         return evals


# class BSplineMultiscaleBasis(Basis):
#     """
#     Parameters
#     ----------
#     domain: array-like, shape = [1, 2]
#         the domain of interest
#     n_basis_1st_scale: int
#         number of basis to consider at the initial scale, even repartition in locs_bounds
#     n_basis_increase: int
#         power of increase of the number of basis at each scale, 2 means for instance double.
#     locs_bounds: array-like, shape = [1, 2]
#         the bounds for the peaks locations of the splines
#     width_init: float
#         Width at the initial scale
#     dilat: float
#         Dilation coefficient, at every scale, the width is divided by this coefficient
#     n_dilat: the number of dilations to perform from the initial scale
#     bounds_disc: bool
#         should the knots outside of the domain be thresholded to account for possible out of domain discontinuity
#     order: int
#         the order of the spline, 3 is for cubic spline for instance.
#     """
#     def __init__(self, domain, n_basis_1st_scale, n_basis_increase, locs_bounds, width_init=1.0, dilat=2.0,
#                  n_dilat=2, bounds_disc=False, order=3, add_constant=True):
#         self.locs_bounds = locs_bounds
#         self.bounds_disc = bounds_disc
#         self.order = order
#         self.add_constant = add_constant
#         self.widths = [width_init * (1 / dilat) ** i for i in range(n_dilat)]
#         n_basis_per_scale = [int(n_basis_1st_scale * n_basis_increase ** i) for i in range(n_dilat)]
#         self.uniscale_bases = [BSplineUniscaleBasis(
#             domain, n_basis_per_scale[i], locs_bounds, width=self.widths[i],
#             bounds_disc=bounds_disc, order=order, add_constant=False) for i in range(n_dilat)]
#         input_dim = 1
#         super().__init__(np.sum(np.array(n_basis_per_scale)) + int(self.add_constant), input_dim, domain)

#     def compute_matrix(self, X):
#         evals = [scale.compute_matrix(X) for scale in self.uniscale_bases]
#         constant = np.ones((X.shape[0], 1))
#         evals.append(constant)
#         return np.concatenate(evals, axis=1)


# class UniscaleCompactlySupported(Basis):
#     """
#     Uniscale basis of compactly supported discrete wavelets

#     Parameters
#     ----------
#     domain: array-like, shape = [1, 2]
#         domain of interest
#     locs_bounds: array_like, shape = [1, 2]
#         bounds for the support of the wavelets considered
#     pywt_name: str, {"coif", "db"}
#         wavelet name
#     moments: int
#         number of vanishing moments
#     dilat: float
#         dilatation coefficient over the undilated mother wavelet
#     translat: float
#         space between each beginning of wavelet
#     approx_level: int
#         approx level to consider in pywt
#     add_constant: bool
#         should the constant function be added
#     """
#     def __init__(self, domain, locs_bounds, pywt_name="coif", moments=3,
#                  dilat=1.0, translat=1.0, approx_level=5, add_constant=True):
#         self.pywt_name = pywt_name + str(moments)
#         self.dilat = dilat
#         self.translat = translat
#         phi, psi, x = pywt.Wavelet(self.pywt_name).wavefun(level=approx_level)
#         x /= self.dilat
#         self.eval_mother = np.sqrt(dilat) * psi
#         trans_grid = []
#         t = locs_bounds[0, 0]
#         while x[-1] + t <= locs_bounds[0, 1]:
#             trans_grid.append(t)
#             t += translat / dilat
#         self.eval_grids = [x + t for t in trans_grid]
#         input_dim = 1
#         self.add_constant = add_constant
#         super().__init__(len(trans_grid) + int(add_constant), input_dim, domain)

#     def compute_matrix(self, X):
#         n = X.shape[0]
#         mat = np.zeros((n, self.n_basis))
#         for i in range(self.n_basis - int(self.add_constant)):
#             mat[:, i] = np.interp(X.squeeze(), self.eval_grids[i], self.eval_mother)
#         if self.add_constant:
#             constant = np.ones((X.shape[0], 1))
#             return np.concatenate((mat, constant), axis=1)
#         else:
#             return mat

#     def get_gram_matrix(self):
#         return np.eye(self.n_basis)


# class MultiscaleCompactlySupported(Basis):
#     """
#     Multiscale basis of compactly supported discrete wavelets

#     Parameters
#     ----------
#     domain: array-like, shape = [1, 2]
#         domain of interest
#     locs_bounds: array_like, shape = [1, 2]
#         bounds for the support of the wavelets considered
#     pywt_name: str, {"coif", "db"}
#         wavelet name
#     moments: int
#         number of vanishing moments
#     init_dilat: float
#         dilatation coefficient over the undilated mother wavelet for the initial scale
#     dilat: float
#         the dilatation coefficients wy which the scale is divided between each scale
#     n_dilat: int
#         nmuber of scales
#     translat: float
#         space between each beginning of wavelet
#     approx_level: int
#         approx level to consider in pywt
#     add_constant: bool
#         should the constant function be added
#     """
#     def __init__(self, domain, locs_bounds, pywt_name="coif", moments=3,
#                  init_dilat=1.0, dilat=2.0, n_dilat=2, translat=1.0,
#                  approx_level=5, add_constant=True):
#         self.scale_bases = []
#         self.add_constant = add_constant
#         for n in range(n_dilat):
#             self.scale_bases.append(UniscaleCompactlySupported(domain, locs_bounds, pywt_name, moments,
#                                                                init_dilat * dilat ** n,
#                                                                translat, approx_level, add_constant=False))
#         n_basis = np.sum([scale.n_basis for scale in self.scale_bases])
#         super().__init__(n_basis + int(self.add_constant), 1, domain)

#     def compute_matrix(self, X):
#         evals = [scale.compute_matrix(X) for scale in self.scale_bases]
#         constant = np.ones((X.shape[0], 1))
#         if self.add_constant:
#             evals.append(constant)
#         return np.concatenate(evals, axis=1)

#     def get_gram_matrix(self):
#         return np.eye(self.n_basis)


# class BasisFromEvals(Basis):

#     def __init__(self, domain, approx_locs, evals, add_constant=False, normalize=True, compute_gram=True):
#         self.approx_locs = approx_locs
#         self.evals = evals.copy()
#         self.add_constant = add_constant
#         if self.add_constant:
#             self.evals = np.concatenate((np.ones((1, self.evals.shape[1])), self.evals))
#         super().__init__(self.evals.shape[0], domain)
#         self.gram_mat = None
#         self.norms = None
#         self.compute_gram = compute_gram
#         self.normalize = normalize
#         if self.normalize:
#             self.compute_norms()
#         if self.compute_gram:
#             self.compute_gram_matrix()

#     def compute_norms(self):
#         mat = self.compute_matrix_nonnormalized(self.approx_locs, np.arange(0, self.n_basis))
#         self.norms = np.sqrt(np.mean(mat ** 2, axis=0))

#     def compute_gram_matrix(self):
#         if self.normalize:
#             mat = self.compute_matrix(self.approx_locs)
#         else:
#             mat = self.compute_matrix_nonnormalized(self.approx_locs, np.arange(0, self.n_basis))
#         self.gram_mat = ((self.domain[0, 1] - self.domain[0, 0])/self.approx_locs.shape[0]) * mat.T.dot(mat)

#     def compute_matrix_nonnormalized(self, X, inds):
#         n = X.shape[0]
#         mat = np.zeros((n, len(inds)))
#         for i in range(len(inds)):
#             mat[:, i] = np.interp(X.squeeze(), self.approx_locs, self.evals[inds[i]])
#         return mat

#     def compute_matrix(self, X):
#         n = X.shape[0]
#         mat = np.zeros((n, self.n_basis))
#         for i in range(self.n_basis):
#             mat[:, i] = np.interp(X.squeeze(), self.approx_locs, self.evals[i])
#         if self.normalize:
#             return np.expand_dims(1 / self.norms, axis=0) * mat
#         else:
#             return mat

#     def get_gram_matrix(self):
#         return self.gram_mat


# # ######################## Data-dependant bases ########################################################################

# class SparseCodingBasis(DataDependantBasis):

#     def __init__(self, n_basis, input_dim, domain, regu, approx_locs, tol=1e-5, max_iter=1000, seed=345):
#         super().__init__(n_basis, input_dim, domain)
#         self.approx_locs = approx_locs
#         self.regu = regu
#         self.seed = seed
#         self.evals = np.zeros((0, approx_locs.shape[0]))
#         self.tol = tol
#         self.max_iter = max_iter
#         self.gram_mat = None

#     def compute_matrix(self, X):
#         n = X.shape[0]
#         mat = np.zeros((n, self.n_basis))
#         for i in range(self.n_basis):
#             mat[:, i] = np.interp(X.squeeze(), self.approx_locs, self.evals[i])
#         return mat

#     def add_atoms(self, evals):
#         if evals.ndim == 1:
#             self.evals = np.concatenate((self.evals, np.expand_dims(evals, axis=0)), axis=0)
#             n_evals = 1
#         else:
#             self.evals = np.concatenate((self.evals, evals), axis=0)
#             n_evals = evals.shape[0]
#         self.n_basis += n_evals

#     def delete_atoms(self, inds):
#         n_delete = len(inds)
#         self.evals = np.delete(self.evals, inds, axis=0)
#         self.n_basis -= n_delete

#     def fit(self, *args):
#         smoother_out = smoothing.LinearInterpSmoother()
#         smoother_out.fit(*args)
#         Yfunc = smoother_out.get_functions()
#         Yeval = np.array([f(self.approx_locs) for f in Yfunc])
#         dict_learn = DictionaryLearning(n_components=self.n_basis, alpha=self.regu,
#                                         tol=self.tol, max_iter=self.max_iter,
#                                         fit_algorithm="cd", random_state=self.seed)
#         dict_learn.fit(Yeval)
#         self.evals = dict_learn.components_
#         self.compute_gram_matrix()

#     def compute_gram_matrix(self, approx_locs=None):
#         if approx_locs is None:
#             space = self.approx_locs
#         else:
#             space = approx_locs
#         mat = self.compute_matrix(space)
#         self.gram_mat = (1/space.shape[0]) * mat.T.dot(mat)

#     def get_gram_matrix(self):
#         return self.gram_mat


# class SparseCodingBasis2(DataDependantBasis):

#     def __init__(self, n_basis, input_dim, domain, regu, approx_locs,
#                  tol=1e-5, max_iter=1000, seed=345, center_output=False):
#         super().__init__(n_basis, input_dim, domain)
#         self.approx_locs = approx_locs
#         self.regu = regu
#         self.evals = None
#         self.tol = tol
#         self.max_iter = max_iter
#         self.gram_mat = None
#         self.seed = seed
#         # TODO: cochon, trouver un fix plus élégant
#         self.center_output = center_output

#     def compute_matrix(self, X):
#         n = X.shape[0]
#         mat = np.zeros((n, self.n_basis))
#         for i in range(self.n_basis):
#             mat[:, i] = np.interp(X.squeeze(), self.approx_locs, self.evals[i])
#         return mat

#     def fit(self, *args):
#         Ymean_func = disc_fd.mean_func(*args)
#         if self.center_output:
#             Ycentered = disc_fd.center_discrete(*args, Ymean_func)
#             Ycentered = disc_fd.to_discrete_general(*Ycentered)
#         else:
#             Ycentered = disc_fd.to_discrete_general(*args)
#         smoother_out = smoothing.LinearInterpSmoother()
#         smoother_out.fit(*Ycentered)
#         Yfunc = smoother_out.get_functions()
#         Yeval = np.array([f(self.approx_locs) for f in Yfunc])
#         dict_learn = DictionaryLearning(n_components=self.n_basis, alpha=self.regu,
#                                         tol=self.tol, max_iter=self.max_iter, n_jobs=-1,
#                                         fit_algorithm="cd", random_state=self.seed)
#         dict_learn.fit(Yeval)
#         self.evals = dict_learn.components_
#         self.compute_gram_matrix()

#     def compute_gram_matrix(self, approx_locs=None):
#         if approx_locs is None:
#             space = self.approx_locs
#         else:
#             space = approx_locs
#         mat = self.compute_matrix(space)
#         self.gram_mat = (1/space.shape[0]) * mat.T.dot(mat)

#     def get_gram_matrix(self):
#         return self.gram_mat


# class FPCABasis(DataDependantBasis):

#     def __init__(self, n_basis, input_dim, domain, n_evals, output_smoother=smoothing.LinearInterpSmoother()):
#         self.fpca = fpca.FunctionalPCA(domain, n_evals, output_smoother)
#         super().__init__(n_basis, input_dim, domain)

#     def fit(self, *args):
#         self.fpca.fit(*args)

#     def compute_matrix(self, X):
#         # evals = [self.fpca.predict(X[i])[:self.n_basis] for i in range(len(X))]
#         # return np.array(evals)
#         n = X.shape[0]
#         funcs = self.fpca.get_regressors(self.n_basis)
#         mat = np.zeros((n, self.n_basis))
#         for i in range(self.n_basis):
#             mat[:, i] = funcs[i](X)
#         return mat

#     def get_gram_matrix(self):
#         return np.eye(self.n_basis)


# # ######################## Basis generation ############################################################################

# SUPPORTED_DICT = {"random_fourier": RandomFourierFeatures,
#                   "fourier": FourierBasis,
#                   "wavelets": MultiscaleCompactlySupported,
#                   "functional_pca": FPCABasis,
#                   "sparse_coding": SparseCodingBasis,
#                   "sparse_coding2": SparseCodingBasis2}


# def generate_basis(key, kwargs):
#     """
#     Generate basis from name and keywords arguments

#     Parameters
#     ----------
#     key: {"random_fourier", "fourier", "wavelets", "from_smooth_funcs", "functional_pca"}
#         The basis reference name
#     kwargs: dict
#         key words argument to build the basis in question

#     Returns
#     -------
#     Basis
#         Generated basis
#     """
#     return SUPPORTED_DICT[key](**kwargs)


# def set_basis_config(passed_basis):
#     """
#     Make the difference between what is passed as basis, can be either a Basis instance or a configuration tuple

#     Parameters
#     ----------
#     passed_basis: Basis or tuple
#         If tuple is given, must be of the form (basis_key, basis_config) with basis_key in `SUPPORTED_DICT` and
#         basis_config a dictionary of basis parameters with the right keys

#     Returns
#     -------
#     tuple,
#         a config and a basis, one of them is None
#     """
#     if isinstance(passed_basis, Basis) or isinstance(passed_basis, DataDependantBasis):
#         actual_basis = passed_basis
#         config_basis = None
#     else:
#         config_basis = passed_basis
#         actual_basis = None
#     return config_basis, actual_basis