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

from regressors import FeaturesKPL, FeaturesKPLOtherLoss
from functional_data import FourierBasis
from optim import AccProxGD
from losses import Huber2Loss
from kernel import GaussianKernel, NystromFeatures
from datasets import add_gp_outliers, load_gp_dataset
from model_selection import tune_features, product_config, test_esti_partial, tune_consecutive, tune
import expe_funcs
import global_config as config


n_averaging = 10

if __name__ == "__main__":
    n_feat = 100
    kerin = GaussianKernel(config.KERNEL_INPUT_GAMMA)
    # lbda_grid = np.geomspace(1e-7, 1e-5, 10)
    lbda_grid = np.geomspace(1e-9, 1e-2, 40)
    # lbda_grid = np.geomspace(1e-7, 1e-5, 2)
    # loss_params = np.flip(np.linspace(0.01, 0.1, 2))
    corrupt_params = config.CORRUPT_GLOBAL_PARAMS
    corrupt_dicts = expe_funcs.interpret_corrupt_params(corrupt_params)
    corrupt_function = add_gp_outliers
    seeds_coefs_train, seeds_coefs_test, seeds_corrupt, seeds_cv = expe_funcs.draw_seeds(
        n_averaging, config.SEED)
    seeds_nys = seeds_coefs_train + 5678
    base_path = str(exec_path.parent.parent.parent)
    out_folder = base_path + "/outputs/results/synth/"
    expe_funcs.create_folder(out_folder)
    fourdict = FourierBasis(0, 40, (0, 1))
    theta = np.linspace(0, 1, 100)
    phi = fourdict.compute_matrix(theta)
    results = np.zeros((n_averaging, len(corrupt_dicts)))
    for i in range(n_averaging):
        nysfeat = NystromFeatures(kerin, n_feat, seeds_nys[i], thresh=0)
        conf = {"regu": lbda_grid, "features": nysfeat, "phi": phi,
                "refit_features": True, "center_out": False}
        # conf = {"regu": torch.logspace(-10, -5, 2), "features": None, "phi": None, "refit_features": False, "center_out": [True, False]}
        confs = product_config(conf, leave_out=["phi"])
        estis = [FeaturesKPL(**params) for params in confs]
        Xtrain, Ytrain, Xtest, Ytest = load_gp_dataset(
            seeds_coefs_train[i], seeds_coefs_test[i])
        Xtrain, Ytrain, Xtest, Ytest = Xtrain.numpy(
        ), Ytrain.numpy(), Xtest.numpy(), Ytest.numpy()
        Ktrain = kerin(Xtrain, Xtrain)
        # Corrupt data
        for j in range(len(corrupt_dicts)):
            Ytrain_corr, _ = corrupt_function(
                torch.from_numpy(Ytrain), Xeval=None, **corrupt_dicts[j], seed=seeds_corrupt[i])
            Ytrain_corr = Ytrain_corr.numpy()
            best_esti, mses = tune(estis, Xtrain, Ytrain_corr, K=Ktrain, Yeval=None,
                                   n_splits=5, reduce_stat="median", random_state=seeds_cv[i], n_jobs=cpu_count() // 5)
            preds = best_esti.predict(Xtest)
            sc = ((preds - Ytest) ** 2).mean()
            results[i, j] = sc
            print(results)
            print("Corrupt param: " + str(j))
        print("Averaging no: " + str(i))
        with open(out_folder + "glob_l2_int_light" + str(i) + ".pkl", "wb") as outp:
            pickle.dump(results, outp)
