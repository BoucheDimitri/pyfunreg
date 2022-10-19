import os
import sys
import pathlib
import torch
import pickle

exec_path = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(str(exec_path.parent.parent))
sys.path.append(str(exec_path.parent))
sys.path.append(str(exec_path))

import expe_funcs
import global_config as config
from kernel import SpeechKernel

# n_averaging = 10
# n_feats = [10, 25, 50, 75, 100, 150, 200, 250, 300]
n_averaging = 2
n_feats = [10, 25, 50]

if __name__ == "__main__":
    stacked_features = [dict() for i in range(n_averaging)]
    seeds_data, seeds_nys, _, seeds_cv = expe_funcs.draw_seeds(
        n_averaging, config.SEED)
    for i in range(n_averaging):
        Xtrain, Ytrain_full_ext, Ytrain_full, Xtest, Ytest_full_ext, Ytest_full = expe_funcs.load_speech_dataset(
            seeds_data[i], str(exec_path.parent.parent) + "/datasets/dataspeech/raw/", n_train=config.N_TRAIN)
        Xtrain = torch.Tensor(Xtrain)
        for n_feat in n_feats:
            # Ks, feats = expe_funcs.pretrain_nystrom_features(Xtrain, n_feat, SpeechKernel, torch.logspace(
            #     *config.KERNEL_INPUT_GAMMA), seeds_cv[i], seeds_nys[i], n_splits=config.CV_SPLIT)
            Ks, feats = expe_funcs.pretrain_nystrom_features(
                Xtrain, n_feat, SpeechKernel, torch.logspace(-2, -0.5, 2), seeds_cv[i], seeds_nys[i], n_splits=config.CV_SPLIT)
            stacked_features[i][n_feat] = (Ks, feats)
            print(n_feat)
    with open(str(exec_path.parent.parent) + "/outputs/pretraining/features_speech.pkl", "wb") as outp:
        pickle.dump(stacked_features, outp)
