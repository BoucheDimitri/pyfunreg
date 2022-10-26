import os
import sys
import pathlib
import torch
import pickle
import numpy as np
from multiprocessing import cpu_count

exec_path = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(str(exec_path.parent.parent.parent))
sys.path.append(str(exec_path.parent.parent))
sys.path.append(str(exec_path.parent))

from regressors import FeaturesKPLWorking
from optim import AccProxGD
from kernel import GaussianKernel, NystromFeatures
from datasets import load_gp_dataset, N_THETA, SyntheticGPmixture
import expe_funcs
import global_config as config


n_averaging = 10

if __name__ == "__main__":
    n_feat = 100
    kerin = GaussianKernel(config.KERNEL_INPUT_GAMMA)
    lbda_grid = np.geomspace(1e-7, 1e-1, 100)
    seeds_coefs_train, seeds_coefs_test, seeds_gps, seeds_cv = expe_funcs.draw_seeds(
        n_averaging, config.SEED)
    seeds_nys = seeds_coefs_train + 5678
    base_path = str(exec_path.parent.parent.parent)
    out_folder = base_path + "/outputs/results/synth/"
    expe_funcs.create_folder(out_folder)
    accproxgd = AccProxGD(n_epoch=20000, stepsize0=1,
                          tol=1e-7, acc_temper=20, verbose=False)
    theta = torch.linspace(0, 1, N_THETA)
    results = np.zeros((n_averaging, len(lbda_grid)))
    working_sets = [[] for i in range(n_averaging)]
    for i in range(n_averaging):
        nysfeat = NystromFeatures(kerin, n_feat, seeds_nys[i], thresh=0)
        Xtrain, Ytrain, Xtest, Ytest = load_gp_dataset(
            seeds_coefs_train[i], seeds_coefs_test[i])
        Xtrain, Ytrain, Xtest, Ytest = Xtrain.numpy(
        ), Ytrain.numpy(), Xtest.numpy(), Ytest.numpy()
        Ktrain = kerin(Xtrain, Xtrain)
        atoms_stds = [0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.5, 0.7]
        n_per_std = 40
        # Parasite atoms
        stds_out = []
        for n, _ in enumerate(atoms_stds):
            stds_out += [atoms_stds[n] for i in range(n_per_std)]
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
        for l, lbda in enumerate(lbda_grid):
            wkpl = FeaturesKPLWorking(1e-9, lbda, nysfeat, phi.numpy(), accproxgd, phi_adj_phi.numpy(), regu_init=1e-9, refit_features=True)
            wkpl.fit(Xtrain, Ytrain, Ktrain)
            preds = wkpl.predict(Xtest)
            sc = ((preds - Ytest) ** 2).mean()
            results[i, l] = sc
            working_sets[i].append(wkpl.working)
            print("Lambda param no: " + str(l))
        print("AVERAGING NO: " + str(i))
        with open(out_folder + "select_descriptive_" + str(i) + ".pkl", "wb") as outp:
            pickle.dump((results, working_sets), outp)
