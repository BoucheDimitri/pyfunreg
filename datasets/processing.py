import numpy as np
import python_speech_features
import copy

from . import loading


# #########################  DTI Dataset ###############################################################################

def set_locs(ylocs, Yobs):
    return [ylocs for i in range(len(Yobs))], Yobs


def process_dti(cca, rcst):
    # Add sampling locations
    locs_cca = np.arange(0, cca.shape[1])
    locs_rcst = np.arange(0, rcst.shape[1])
    rcst_interp = copy.copy(rcst)
    # Use interpolation for filling NaNs in the input data if relevant
    for i in range(len(cca)):
        cca[i] = np.interp(locs_cca,
                           locs_cca[np.argwhere(~ np.isnan(cca[i])).squeeze()],
                           cca[i][np.argwhere(~ np.isnan(cca[i])).squeeze()])
    for i in range(len(cca)):
        rcst_interp[i] = np.interp(locs_rcst,
                                   locs_rcst[np.argwhere(~ np.isnan(rcst_interp[i])).squeeze()],
                                   rcst_interp[i][np.argwhere(~ np.isnan(rcst_interp[i])).squeeze()])
    return cca, rcst, rcst_interp



# #########################  Speech dataset ############################################################################

RATE = 10000
OUTPUT_PACE = 0.005


def normalize_domain_max_duration(X):
    max_duration = np.max([x.shape[0] / RATE for x in X])
    norma = (1 / (max_duration - OUTPUT_PACE))
    return norma


def extend_max_duration_input(x, length, max_length):
    n_signal_pad = int((max_length / length))
    xpad = np.pad(x, pad_width=(0, n_signal_pad * x.shape[0]), mode="symmetric")
    xpad = xpad[:int(max_length * 10000)]
    return xpad


def extend_max_duration_outputs(y, length, max_length):
    ylocs = np.arange(0, max_length - OUTPUT_PACE, OUTPUT_PACE)
    locs_size = len(ylocs.squeeze())
    n_signal_pad = int((max_length / length))
    ypad = np.pad(y, pad_width=(0, n_signal_pad * len(y)), mode="symmetric")
    return ylocs, ypad[:locs_size]


def append_processed_data_point(X, Y, Xout, Yout, i, normalize_domain, extend):
    # Signal length
    length = X[i].shape[0] / RATE
    # Maximum duration of signals
    max_length = np.max([sig.shape[0] / RATE for sig in X])
    # Locations of sampling of the output function
    norma = normalize_domain_max_duration(X) if normalize_domain else 1
    # Pad the signal if necessary
    xpad = extend_max_duration_input(X[i], length, max_length) if extend else X[i]
    # Compute MFCCs
    mfccs = python_speech_features.base.mfcc(xpad, samplerate=RATE, winlen=0.010, winstep=OUTPUT_PACE, numcep=13)
    Xout.append(mfccs)
    for vt in loading.VOCAL_TRACTS:
        if extend:
            ylocs, ypad = extend_max_duration_outputs(Y[vt][i], length, max_length)
            ylocs *= norma
        else:
            ypad = Y[vt][i]
            ylocs = norma * np.arange(0, length - OUTPUT_PACE, OUTPUT_PACE)
        Yout[vt][0].append(ylocs)
        Yout[vt][1].append(ypad)


def normalize_output_values(Ytrain, Ytest):
    Ytrain_norm = {vt: [list(), list()] for vt in loading.VOCAL_TRACTS}
    Ytest_norm = {vt: [list(), list()] for vt in loading.VOCAL_TRACTS}
    norm_vals = dict()
    n_train = len(Ytrain[loading.VOCAL_TRACTS[0]][1])
    n_test = len(Ytest[loading.VOCAL_TRACTS[0]][1])
    for vt in loading.VOCAL_TRACTS:
        m = np.nanmin(np.concatenate([Ytrain[vt][1][i] for i in range(n_train)]))
        M = np.nanmax(np.concatenate([Ytrain[vt][1][i] for i in range(n_train)]))
        a = 2 / (M - m)
        b = 1 - a * M
        norm_vals[vt] = (a, b)
    for i in range(n_train):
        for vt in loading.VOCAL_TRACTS:
            Ytrain_norm[vt][1].append(norm_vals[vt][0] * Ytrain[vt][1][i] + norm_vals[vt][1])
            Ytrain_norm[vt][0].append(Ytrain[vt][0][i])
    for i in range(n_test):
        for vt in loading.VOCAL_TRACTS:
            Ytest_norm[vt][1].append(norm_vals[vt][0] * Ytest[vt][1][i] + norm_vals[vt][1])
            Ytest_norm[vt][0].append(Ytest[vt][0][i])
    return Ytrain_norm, Ytest_norm


def process_speech(X, Y, shuffle_seed=784, n_train=300, normalize_domain=True, normalize_values=True):
    # Initialize containers
    Xtrain_ext, Xtrain, Xtest_ext, Xtest = list(), list(), list(), list()
    Ytrain_ext = {vt: [list(), list()] for vt in loading.VOCAL_TRACTS}
    Ytrain = {vt: [list(), list()] for vt in loading.VOCAL_TRACTS}
    Ytest_ext = {vt: [list(), list()] for vt in loading.VOCAL_TRACTS}
    Ytest = {vt: [list(), list()] for vt in loading.VOCAL_TRACTS}
    # Shuffle index
    n_samples = len(X)
    inds = np.arange(n_samples)
    if shuffle_seed is not None:
        np.random.seed(shuffle_seed)
        np.random.shuffle(inds)
    # Fill data
    for i in inds[:n_train]:
        append_processed_data_point(X, Y, Xtrain_ext, Ytrain_ext, i, normalize_domain, extend=True)
        append_processed_data_point(X, Y, Xtrain, Ytrain, i, normalize_domain, extend=False)
    for i in inds[n_train:]:
        append_processed_data_point(X, Y, Xtest_ext, Ytest_ext, i, normalize_domain, extend=True)
        append_processed_data_point(X, Y, Xtest, Ytest, i, normalize_domain, extend=False)
    # Normalize output values if necessary
    if normalize_values:
        # Extended or not, this results in the same normalization because of extended is a repetition of the same signal
        Ytrain_ext_norm, Ytest_ext_norm = normalize_output_values(Ytrain_ext, Ytest_ext)
        Ytrain_norm, Ytest_norm = normalize_output_values(Ytrain, Ytest)
        return Xtrain_ext, Ytrain_ext_norm, Ytrain_norm, Xtest_ext, Ytest_ext_norm, Ytest_norm
    else:
        return Xtrain_ext, Ytrain_ext, Ytrain, Xtest_ext, Ytest_ext, Ytest
