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


n_feats = [10, 25, 50, 75, 100, 150, 200, 250, 300]


with open(path + "results/large_scale_speech/GLO.pkl", "rb") as inp:
    results_dict = pickle.load(inp)


means_dict = dict()
stds_dict = dict()

keys = ("LA", "LP", "TTCD", "TTCL", "TBCD", "TBCL", "VEL", "GLO")

for key in keys:
    means_dict[key] = torch.tensor(results_dict[key]).mean(dim=0)
    stds_dict[key] = torch.tensor(results_dict[key]).std(dim=0)

fig, axes = plt.subplots(2, 4)


for i, key in enumerate(keys):
    axes[i // 4, i % 4].errorbar(n_feats, means_dict[key], yerr=stds_dict[key], marker=None, capsize=5)
    axes[i // 4, i % 4].set_title(key)

axes[0, 0].set_ylabel("MSE")
axes[1, 0].set_ylabel("MSE")
axes[1, 0].set_xlabel("Number of Nyström features")
axes[1, 1].set_xlabel("Number of Nyström features")
axes[1, 2].set_xlabel("Number of Nyström features")
axes[1, 3].set_xlabel("Number of Nyström features")
plt.show()

