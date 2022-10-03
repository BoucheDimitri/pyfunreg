import os
import sys

fontsize = 30
pad = 20

sys.path.append(os.getcwd())
from datasets import load_raw_speech, process_speech

seed = 1454
n_train = 300

X, Y = load_raw_speech(str(os.getcwd()) + "/datasets/dataspeech/raw/")
Xtrain, Ytrain_full_ext, Ytrain_full, Xtest, Ytest_full_ext, Ytest_full = process_speech(X, Y, shuffle_seed=seed, n_train=n_train)