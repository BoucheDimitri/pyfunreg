import pandas as pd
import numpy as np
import os
from scipy.io import wavfile


def load_dti(path, shuffle_seed=0):
    # Set random seed
    np.random.seed(shuffle_seed)
    # Load the data
    cca = pd.read_csv(path + "cca_DTI_1st_visit.csv").iloc[:, 1:]
    rcst = pd.read_csv(path + "rcst_DTI_1st_visit.csv").iloc[:, 1:]
    n = cca.shape[0]
    # Shuffle index
    shuffle_inds = np.random.choice(n, n, replace=False)
    # Extract in numpy form and apply shuffle
    try:
        cca = cca.to_numpy()[shuffle_inds, 0:]
        rcst = rcst.to_numpy()[shuffle_inds, 0:]
    except AttributeError:
        cca = cca.values[shuffle_inds, 0:]
        rcst = rcst.values[shuffle_inds, 0:]
    return cca, rcst


VOCAL_TRACTS = ("LP", "LA", "TBCL", "TBCD", "VEL", "GLO", "TTCL", "TTCD")


def load_raw_speech(path):
    words = list(set([word.split(".")[0] for word in os.listdir(path)]))
    words.sort()
    X = []
    Y = {vt: list() for vt in VOCAL_TRACTS}
    for word in words:
        rate, signal = wavfile.read(path + word + ".wav")
        output_func_mat = pd.read_csv(path + word + ".csv", header=None)
        X.append(signal)
        try:
            output_func_mat = output_func_mat.to_numpy()
        except AttributeError:
            output_func_mat = output_func_mat.values
        for i in range(len(VOCAL_TRACTS)):
            Y[VOCAL_TRACTS[i]].append(output_func_mat[i])
    return X, Y
