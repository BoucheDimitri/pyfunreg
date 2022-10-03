import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl


fontsize = 30
pad = 20

sys.path.append(os.getcwd())
from datasets.outliers import add_gp_outliers
from datasets import load_gp_dataset


torch.set_default_dtype(torch.float64)

seeds_coefs_train = np.random.choice(np.arange(100, 100000), 10, replace=False)
seeds_coefs_test = np.random.choice(np.arange(100, 100000),10, replace=False)
theta = torch.linspace(0, 1, 100)


# ############################ EXAMPLES WITH OUTLIERS ##############################################
Xtrain, Ytrain, Xtest, Ytest = load_gp_dataset(seeds_coefs_train[0], seeds_coefs_test[0])
Ytrain_corr, _ = add_gp_outliers(Ytrain, freq_sample=1., intensity=2.)


n_plots = 50
colors = [cm.viridis(x) for x in np.linspace(0, 1, n_plots)]

fig, axes = plt.subplots(3, 1, sharex="col")
for i in range(n_plots):
    axes[0].plot(theta, Xtrain[i], c=colors[i])
    axes[1].plot(theta, Ytrain[i], c=colors[i])
    axes[2].plot(theta, Ytrain_corr[i], c=colors[i])
axes[0].set_title("Input functions $(x_i)_{i=1}^{50}$", fontsize=fontsize, pad=pad)
axes[1].set_title("Output functions $(y_i)_{i=1}^{50}$", fontsize=fontsize, pad=pad)
axes[2].set_title("Output functions outliers ($\zeta=2$)", fontsize=fontsize, pad=pad)
plt.show()