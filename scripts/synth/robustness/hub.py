import os
import sys
import pathlib
import torch
import pickle

exec_path = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(str(exec_path.parent.parent.parent))
sys.path.append(str(exec_path.parent.parent))
sys.path.append(str(exec_path.parent))

import global_config as config
import expe_funcs
from model_selection import tune_features, product_config, test_esti_partial, tune_consecutive
from regressors import FeaturesKPL ,FeaturesKPLOtherLoss
from datasets import add_gp_outliers, load_gp_dataset
from kernel import GaussianKernel, NystromFeatures
from losses import Huber2Loss
from optim import AccProxGD
from functional_data import FourierBasis

n_averaging = 2

if __name__ == "__main__":
    n_feat = 150
    kerin = GaussianKernel(config.KERNEL_INPUT_GAMMA)
    lbda_grid = torch.logspace(-9, -5, 10)
    loss_params = torch.linspace(0.01, 0.1, 20)
    losses = [Huber2Loss(param) for param in loss_params]
    corrupt_params = config.CORRUPT_GLOBAL_FREQ_PARAMS
    corrupt_dicts = expe_funcs.interpret_corrupt_params(corrupt_params)
    corrupt_function = add_gp_outliers
    seeds_coefs_train, seeds_coefs_test, seeds_corrupt, seeds_cv = expe_funcs.draw_seeds(n_averaging, config.SEED)
    base_path = str(exec_path.parent.parent.parent)
    out_folder = base_path + "/outputs/results/robustness_synth/"
    expe_funcs.create_folder(out_folder)
    accproxgd = AccProxGD(n_epoch=20000, stepsize0=1, tol=1e-6, acc_temper=20, verbose=False)
    fourdict = FourierBasis(0, 40, (0, 1))
    theta = torch.linspace(0, 1, 100)
    phi = torch.from_numpy(fourdict.compute_matrix(theta.numpy()))
    results = torch.zeros((n_averaging, len(corrupt_dicts)))
    for i in range(n_averaging):
        nysfeat = NystromFeatures(kerin, n_feat, 432)
        conf = {"regu": lbda_grid, "loss": None, "features": nysfeat, "phi": phi, "refit_features": True, "center_out": False, "optimizer": accproxgd}
        # conf = {"regu": torch.logspace(-10, -5, 2), "features": None, "phi": None, "refit_features": False, "center_out": [True, False]}
        confs = product_config(conf, leave_out=["phi"])
        estis = [FeaturesKPLOtherLoss(**params) for params in confs]
        Xtrain, Ytrain, Xtest, Ytest = load_gp_dataset(seeds_coefs_train[i], seeds_coefs_test[i])
        Ktrain = kerin(Xtrain, Xtrain)
        # Corrupt data
        for j in range(len(corrupt_dicts)):
            Ytrain_corr, _ = corrupt_function(
                Ytrain, Xeval=None, **corrupt_dicts[i], seed=seeds_corrupt[i])
            best_esti, mses = tune_consecutive(estis, losses, Xtrain, Ytrain, K=Ktrain, Yeval=None, n_splits=5, reduce_stat="median", random_state=seeds_cv[i], n_jobs=-1)
            preds = best_esti.predict(Xtest)
            sc = ((preds - Ytest) ** 2).mean()
            results[i, j] = sc
        print(i)
        with open(out_folder + "global_hub2" + str(i) + ".pkl", "wb") as outp:
            pickle.dump(results, outp)