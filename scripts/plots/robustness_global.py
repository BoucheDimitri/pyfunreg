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


with open(path + "results/synth/glob_hub2_freq3.pkl", "rb") as inp:
    results = pickle.load(inp)

means = results[[0, 2, 3]].mean(axis=0)
meds = np.quantile(results[:4], q=0.5, axis=0)