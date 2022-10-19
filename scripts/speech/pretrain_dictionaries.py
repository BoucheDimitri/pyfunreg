import os
import sys
import pathlib
import torch
import pickle

exec_path = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(str(exec_path.parent.parent))
sys.path.append(str(exec_path.parent))
sys.path.append(str(exec_path))

import global_config as config
import expe_funcs

# n_averaging = 10
n_averaging = 2

if __name__ == "__main__":
    stacked_dicts = [dict() for i in range(n_averaging)]
    seeds_data, seeds_dict, _, seeds_cv = expe_funcs.draw_seeds(
        n_averaging, config.SEED)
    for i in range(n_averaging):
        Xtrain, Ytrain_full_ext, Ytrain_full, Xtest, Ytest_full_ext, Ytest_full = expe_funcs.load_speech_dataset(
            seeds_data[i], str(exec_path.parent.parent) + "/datasets/dataspeech/raw/", n_train=config.N_TRAIN)
        Xtrain = torch.Tensor(Xtrain)
        for key in config.KEYS:
            y = torch.Tensor(Ytrain_full_ext[key][1])
            phis = expe_funcs.pretrain_dictionaries(y, seeds_cv[i], n_splits=5, n_components=30, alpha=1e-5, tol=1e-5, max_iter=5000, dl_seed=seeds_dict[i])
            stacked_dicts[i][key] = phis
            print(key)
    with open(str(exec_path.parent.parent) + "/outputs/pretraining/dicts_speech.pkl", "wb") as outp:
        pickle.dump(stacked_dicts, outp)

