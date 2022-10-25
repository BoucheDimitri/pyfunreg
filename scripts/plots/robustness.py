import sys
import os
import pickle
import torch
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
from matplotlib import gridspec
import scipy.stats as stats

# Plot parameters
plt.rcParams.update({"pdf.fonttype": 42})
plt.rcParams.update({"font.size": 27})
# plt.rcParams.update({'ps.useafm': True})
plt.rcParams.update({"lines.linewidth": 4})
plt.rcParams.update({"lines.markersize": 10})
plt.rcParams.update({"axes.linewidth": 2})
plt.rcParams.update({"xtick.major.size": 5})
plt.rcParams.update({"xtick.major.width": 1.5})
plt.rcParams.update({"ytick.major.size": 5})
plt.rcParams.update({"ytick.major.width": 1.5})

cwd = pathlib.Path(os.getcwd()) / "scripts" / "plot"
sys.path.append(str(cwd))

import tsi_cluster.cluster_outputs as cluster_outputs

path = cluster_outputs.path_to_latest() + "/outputs/"


exp_keys = ["glob_hub2_freq", "glob_hubinf_freq", "glob_hub2_int", "glob_hubinf_int", "loc_hub2_freq", "loc_hubinf_freq", "loc_hub2_int", "loc_hubinf_int"]
n_avgs = [10, 8, 10, 8, 10, 7, 10, 10]
meds = dict()
mads = dict()

for i, key in enumerate(exp_keys):
    with open(path + "results/synth/" + key + "_light" + str(n_avgs[i] - 1) + ".pkl", "rb") as inp:
        res = pickle.load(inp)
    meds[key] = np.quantile(res[:n_avgs[i]], q=0.5, axis=0)
    mads[key] = stats.median_abs_deviation(res[:n_avgs[i]], axis=0)



