import global_config
import expe_funcs
import os
import sys
import pathlib


exec_path = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(str(exec_path.parent.parent))
sys.path.append(str(exec_path.parent))
sys.path.append(str(exec_path))


n_averaging = 10

if __name__ == "__main__":
    seeds_data, _, _, _, seeds_cv = expe_funcs.draw_seeds(
        n_averaging, global_config.SEED)
    for i in range(n_averaging):
        Xtrain, Ytrain_full_ext, Ytrain_full, Xtest, Ytest_full_ext, Ytest_full = expe_funcs.load_speech_dataset(
            seeds_data[i], str(exec_path.parent.parent) + "/datasets/dataspeech/raw/", n_train=global_config.N_TRAIN)
        
    
