import os
import sys
import pathlib
import torch
import pickle
import numpy as np
from multiprocessing import cpu_count
import time

exec_path = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(str(exec_path.parent.parent.parent))
sys.path.append(str(exec_path.parent.parent))
sys.path.append(str(exec_path.parent))

from regressors import FeaturesKPL
from optim import AccProxGD
from kernel import GaussianKernel, NystromFeatures
from datasets import load_gp_dataset, N_THETA, SyntheticGPmixture
import expe_funcs
import global_config as config
from model_selection import product_config, tune


if __name__ == "__main__":
    n_averaging = 10
    n_feat = 100
    thresh = 1e-1
    ns_per_std = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
    # ns_per_std = [1, 5, 10]
    kerin = GaussianKernel(config.KERNEL_INPUT_GAMMA)
    lbda_grid = np.geomspace(1e-8, 1e-2, 50)
    seeds_coefs_train, seeds_coefs_test, seeds_gps, seeds_cv = expe_funcs.draw_seeds(
        n_averaging, config.SEED)
    seeds_nys = seeds_coefs_train + 5678
    base_path = str(exec_path.parent.parent.parent)
    out_folder = base_path + "/outputs/results/synth/"
    expe_funcs.create_folder(out_folder)
    theta = torch.linspace(0, 1, N_THETA)
    results = np.zeros((n_averaging, len(ns_per_std)))
    results_thresh = np.zeros((n_averaging, len(ns_per_std)))
    timers_svd = np.zeros((n_averaging, len(ns_per_std)))
    timers_fit = np.zeros((n_averaging, len(ns_per_std)))
    timers_tfit = np.zeros((n_averaging, len(ns_per_std)))
    for i in range(n_averaging):
        nysfeat = NystromFeatures(kerin, n_feat, seeds_nys[i], thresh=0)
        Xtrain, Ytrain, Xtest, Ytest = load_gp_dataset(
            seeds_coefs_train[i], seeds_coefs_test[i])
        Xtrain, Ytrain, Xtest, Ytest = Xtrain.numpy(
        ), Ytrain.numpy(), Xtest.numpy(), Ytest.numpy()
        Ktrain = kerin(Xtrain, Xtrain)
        for p, d in enumerate(ns_per_std):
            atoms_stds = [0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.5, 0.7]
            # Parasite atoms
            stds_out = []
            for n, _ in enumerate(atoms_stds):
                stds_out += [atoms_stds[n] for i in range(d)]
            stds_in = stds_out
            scale = 1.5
            n_atoms = len(stds_in)
            gamma_cov = torch.Tensor([stds_in, stds_out]).numpy()
            data_gp = SyntheticGPmixture(n_atoms=n_atoms, gamma_cov=gamma_cov, scale=scale)
            data_gp.drawGP(theta, seed_gp=seeds_gps[i])
            big_dict = data_gp.GP_output
            phi = big_dict.T
            m = len(theta)
            gram_mat = (1 / m) * phi.T @ phi
            phi *= torch.sqrt((1 / torch.diag(gram_mat).unsqueeze(0)))
            phi_adj_phi = (1 / m) * phi.T @ phi
            conf = {"regu": lbda_grid, "features": nysfeat, "phi": phi.numpy(), "phi_adj_phi": phi_adj_phi.numpy(),
                    "refit_features": True, "center_out": False}
            # conf = {"regu": torch.logspace(-10, -5, 2), "features": None, "phi": None, "refit_features": False, "center_out": [True, False]}
            confs = product_config(conf, leave_out=["phi", "phi_adj_phi"])
            estis = [FeaturesKPL(**params) for params in confs]
            start = time.process_time()
            u, V = np.linalg.eigh(phi_adj_phi)
            end = time.process_time()
            timers_svd[i, p] = end - start
            uthresh = u[u > thresh]
            Vthresh = V[:, u > thresh]
            phi_thresh = phi @ Vthresh
            phi_adj_phi_thresh = np.diag(uthresh)
            conf_thresh = {"regu": lbda_grid, "features": nysfeat, "phi": phi_thresh.numpy(), "phi_adj_phi": np.diag(uthresh),
            "refit_features": True, "center_out": False}
            # conf = {"regu": torch.logspace(-10, -5, 2), "features": None, "phi": None, "refit_features": False, "center_out": [True, False]}
            confs_thresh = product_config(conf_thresh, leave_out=["phi", "phi_adj_phi"])
            estis_thresh = [FeaturesKPL(**params) for params in confs_thresh]
            best_esti, mses = tune(estis, Xtrain, Ytrain, K=Ktrain, Yeval=None,
                                   n_splits=5, reduce_stat="mean", random_state=seeds_cv[i], n_jobs=1)
            timers_fit[i, p] = best_esti.fit(Xtrain, Ytrain, Ktrain, return_timer=True)
            preds = best_esti.predict(Xtest)
            sc = ((preds - Ytest) ** 2).mean()
            results[i, p] = sc
            best_esti_thresh, mses = tune(estis_thresh, Xtrain, Ytrain, K=Ktrain, Yeval=None,
                                   n_splits=5, reduce_stat="mean", random_state=seeds_cv[i], n_jobs=1)
            timers_tfit[i, p] = best_esti_thresh.fit(Xtrain, Ytrain, Ktrain, return_timer=True)
            preds = best_esti_thresh.predict(Xtest)
            sc_thresh = ((preds - Ytest) ** 2).mean()
            results_thresh[i, p] = sc_thresh
            print("N DICT: " + str(d))
        print("AVERAGING NO: " + str(i))
        with open(out_folder + "lowrank_descriptive_" + str(i) + ".pkl", "wb") as outp:
            pickle.dump((timers_svd, results, results_thresh, timers_fit, timers_tfit), outp)
