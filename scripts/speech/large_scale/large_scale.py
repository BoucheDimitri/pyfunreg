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
from model_selection import tune_features, product_config, test_esti_partial
from regressors import FeaturesKPL

n_averaging = 10
n_feats = [10, 25, 50, 75, 100, 150, 200, 250, 300]
lbda_search = torch.logspace(-10, -5, 20)


if __name__ == "__main__":
    results_dict = {key: torch.zeros((n_averaging, len(n_feats))) for key in config.KEYS}
    # For some reason this does not bug if we loop over keys first...
    for key in config.KEYS:
        base_path = str(exec_path.parent.parent.parent)
        out_folder = base_path + "/outputs/results/large_scale_speech/"
        expe_funcs.create_folder(out_folder)
        conf = {"regu": lbda_search, "features": None, "phi": None, "refit_features": False, "center_out": [True, False]}
        # conf = {"regu": torch.logspace(-10, -5, 2), "features": None, "phi": None, "refit_features": False, "center_out": [True, False]}
        confs = product_config(conf, leave_out=["phi"])
        estis = [FeaturesKPL(**params) for params in confs]
        with open(base_path + "/outputs/pretraining/dicts_speech.pkl", "rb") as inp:
            stacked_dicts = pickle.load(inp)
        with open(base_path + "/outputs/pretraining/features_speech.pkl", "rb") as inp:
            stacked_features = pickle.load(inp)
        seeds_data, seeds_dict, _, seeds_cv = expe_funcs.draw_seeds(
            n_averaging, config.SEED)
        for i in range(n_averaging):
            Xtrain, Ytrain_full_ext, Ytrain_full, Xtest, Ytest_full_ext, Ytest_full = expe_funcs.load_speech_dataset(
                seeds_data[i], base_path + "/datasets/dataspeech/raw/", n_train=config.N_TRAIN)
            Xtrain = torch.Tensor(Xtrain)
            Xtest = torch.Tensor(Xtest)
            ytrain = torch.Tensor(Ytrain_full_ext[key][1])
            for l, n_feat in enumerate(n_feats):
                Ks, feats =  stacked_features[i][n_feat]
                best_esti, mses = tune_features(estis, feats, Xtrain, ytrain, Ks=Ks, 
                    Yeval=Ytrain_full[key][1], phis=stacked_dicts[i][key][0], phi_test=stacked_dicts[i][key][1], n_splits=5, reduce_stat="mean", 
                    random_state=seeds_cv[i], n_jobs=-1)
                sc = test_esti_partial(best_esti, Xtest, Ytest_full[key][1])
                results_dict[key][i, l] = sc
        print(key)
        print(results_dict[key])
        with open(out_folder + str(key) + ".pkl", "wb") as outp:
            pickle.dump(results_dict, outp)
    print(results_dict)

