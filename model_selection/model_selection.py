from typing import Iterable
import torch
import numpy as np
from sklearn.model_selection import KFold
import os
from joblib import delayed, Parallel, parallel_backend
from multiprocessing import cpu_count



def create_folder(folder):
    folder_split = str(folder).split("/")
    path = ""
    for fold in folder_split:
        path += "/" + fold
        try:
            os.mkdir(path)
        except FileExistsError:
            pass


def cv_esti_iter_consecutive(esti, loss_params, X, Y, thetas, Kx=None,
                             Yeval=None, n_splits=5, reduce_stat="median",
                             random_state=342,
                             n_epoch=10000, warm_start=True, tol=1e-8, beta=0.8, monitor_loss=None, d=20):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    mses = torch.zeros((5, len(loss_params)))
    count = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        if G_x is not None:
            G_x_train = G_x[train_index, :][:, train_index]
            G_x_test = G_x[test_index, :][:, train_index]
        else:
            G_x_train, G_x_test = None, None
        if esti_l2_init is not None:
            esti_l2_init.lbda = esti.lbda
            esti_l2_init.fit(X_train, Y_train, G_x_train, G_t)
            esti.alpha = esti_l2_init.alpha.detach().clone()
        for i, loss_param in enumerate(loss_params):
            esti.loss_param = loss_param
            fit_esti_iter(esti, X_train, Y_train, thetas, G_x_train, G_t, solver, n_epoch, warm_start, tol,
                          beta, monitor_loss, d, return_esti=False)
            pred_test = esti.model.forward(X_test, G_x_test)
            if Yeval is not None:
                mses[count, i] = get_score(pred_test, Yeval[test_index], partial=True, metric_p=metric_p)
            else:
                mses[count, i] = get_score(pred_test, Y_test, partial=False, metric_p=metric_p)
        print(count)
        count += 1
    if reduce_stat == "mean":
        return torch.tensor(mses).mean(dim=0)
    else:
        return torch.tensor(mses).quantile(0.5, dim=0)
