import torch
import numpy as np


def acc_proxgd_lsearch(prox, obj, t0, alpha_v, grad_v, beta=0.2):
    t = t0
    stop = False
    while not stop:
        alpha_plus = prox(alpha_v - t * grad_v, t)
        term1 = obj(alpha_plus)
        term21 = obj(alpha_v)
        term22 = (grad_v * (alpha_plus - alpha_v)).sum()
        term23 = 0.5 * (1 / t) * ((alpha_plus - alpha_v) ** 2).sum()
        term2 = term21 + term22 + term23
        if term1 > term2:
            t *= beta
        else:
            stop = True
    return t


def acc_proxgd(alpha0, prox, obj, obj_full, grad, n_epoch=20000, tol=1e-6, beta=0.8, acc_temper=20, monitor=None, stepsize0=0.1):
    alpha_minus1 = alpha0
    alpha_minus2 = alpha0
    step_size = stepsize0
    converged = False
    monitored = []
    for epoch in range(0, n_epoch):
        acc_cste = epoch / (epoch + 1 + acc_temper)
        alpha_v = alpha_minus1 + acc_cste * (alpha_minus1 - alpha_minus2)
        grad_v = grad(alpha_v)
        step_size = acc_proxgd_lsearch(
            prox, obj, step_size, alpha_v, grad_v, beta)
        alpha = prox(alpha_v - step_size * grad_v, step_size)
        if monitor is not None:
            monitored.append(monitor(alpha))
        # if alpha_minus1.norm() < 1e-10:
        #     raise ValueError("Norm too small")
        # diff = (alpha - alpha_minus1).norm() / alpha_minus1.norm()
        # diff = (obj_full(alpha) - obj_full(alpha_minus1)).abs()
        # print(diff)
        # if diff < tol:
        #     converged = True
        #     break
        print(obj_full(alpha).item())
        alpha_minus2 = alpha_minus1.detach().clone()
        alpha_minus1 = alpha.detach().clone()
    return alpha, monitored


# def acc_proxgd(alpha0, prox, obj, grad, step_size, n_epoch=20000, tol=1e-6, beta=0.8, d=20, monitor=None):
#     alpha_minus1 = alpha0
#     alpha_minus2 = alpha0
#     step_size = 1
#     converged = False
#     monitored = []
#     for epoch in range(0, n_epoch):
#         acc_cste = epoch / (epoch + 1 + d)
#         alpha_v = alpha_minus1 + acc_cste * (alpha_minus1 - alpha_minus2)
#         grad_v = grad(alpha_v)
#         # step_size = acc_proxgd_lsearch(
#         #     prox, obj, step_size, alpha_v, grad_v, beta)
#         alpha = prox(alpha_v - step_size * grad_v, step_size)
#         if monitor is not None:
#             monitored.append(monitor(alpha))
#         if alpha_minus1.norm() < 1e-10:
#             raise ValueError("Norm too small")
#         diff = (alpha - alpha_minus1).norm() / alpha_minus1.norm()
#         print(diff)
#         if diff < tol:
#             converged = True
#             break
#         alpha_minus2 = alpha_minus1.detach().clone()
#         alpha_minus1 = alpha.detach().clone()
#     return alpha, monitored


class AccProxGD:

    def __init__(self, n_epoch=20000, tol=1e-5, beta=0.8, acc_temper=20, monitor="obj", stepsize0=0.1, verbose=True):
        self.n_epoch = n_epoch
        self.tol = tol
        self.beta = beta
        self.acc_temper = acc_temper
        self.monitor = monitor
        self.stepsize0 = stepsize0
        self.verbose = verbose

    def __call__(self, alpha0, prox, obj, obj_full, grad):
        alpha_minus1 = alpha0
        alpha_minus2 = alpha0
        step_size = self.stepsize0
        epoch_restart = 0
        converged = False
        monitored = []
        for epoch in range(0, self.n_epoch):
            acc_cste = epoch_restart / (epoch_restart + 1 + self.acc_temper)
            alpha_v = alpha_minus1 + acc_cste * (alpha_minus1 - alpha_minus2)
            grad_v = grad(alpha_v)
            step_size = acc_proxgd_lsearch(
                prox, obj, step_size, alpha_v, grad_v, self.beta)
            alpha_tentative = prox(
                alpha_v - step_size * grad_v, step_size)
            if ((alpha_v - alpha_tentative) * (alpha_tentative - alpha_minus1)).sum() > 0:
                print("RESTART")
                epoch_restart = 0
                grad_v = grad(alpha_minus1)
                step_size = acc_proxgd_lsearch(
                    prox, obj, step_size, alpha_minus1, grad_v, self.beta)
                alpha = prox(alpha_minus1 - step_size * grad_v, step_size)
                print("Stepsize: " + str(step_size))
            else:
                alpha = alpha_tentative
            if self.monitor == "obj":
                monitored.append(obj(alpha))
            elif self.monitor == "obj_full":
                monitored.append(obj_full(alpha))
            if isinstance(alpha, torch.Tensor):
                diff = (obj_full(alpha) - obj_full(alpha_minus1)
                        ).abs() / obj_full(alpha_minus1).abs()
                if self.verbose:
                    print(diff)
                if diff.item() < self.tol:
                    converged = True
                    break
                alpha_minus2 = alpha_minus1.detach().clone()
                alpha_minus1 = alpha.detach().clone()
            else:
                diff =  np.abs(obj_full(alpha) - obj_full(alpha_minus1)) / np.abs(obj_full(alpha_minus1))
                if self.verbose:
                    print(diff)
                if diff < self.tol:
                    converged = True
                    break
                alpha_minus2 = np.copy(alpha_minus1)
                alpha_minus1 = np.copy(alpha)
            epoch_restart += 1
        return alpha, monitored
        # if not converged:
        #     raise ConvergenceWarning("Maximum number of iteration reached")


def acc_proxgd_restart(alpha0, prox, obj, obj_full, grad, n_epoch=20000, tol=1e-6,
                       beta=0.8, acc_temper=20, monitor=None, stepsize0=0.1, verbose=True):
    alpha_minus1 = alpha0
    alpha_minus2 = alpha0
    step_size = stepsize0
    epoch_restart = 0
    converged = False
    monitored = []
    for epoch in range(0, n_epoch):
        acc_cste = epoch_restart / (epoch_restart + 1 + acc_temper)
        alpha_v = alpha_minus1 + acc_cste * (alpha_minus1 - alpha_minus2)
        grad_v = grad(alpha_v)
        step_size = acc_proxgd_lsearch(
            prox, obj, step_size, alpha_v, grad_v, beta)
        alpha_tentative = prox(
            alpha_v - step_size * grad_v, step_size)
        if ((alpha_v - alpha_tentative) * (alpha_tentative - alpha_minus1)).sum() > 0:
            print("RESTART")
            epoch_restart = 0
            grad_v = grad(alpha_minus1)
            step_size = acc_proxgd_lsearch(
                prox, obj, step_size, alpha_minus1, grad_v, beta)
            alpha = prox(alpha_minus1 - step_size * grad_v, step_size)
            print("Stepsize: " + str(step_size))
        else:
            alpha = alpha_tentative
        if monitor is not None:
            monitored.append(monitor(alpha))
        # diff = (alpha - alpha_minus1).norm() / alpha_minus1.norm()
        # if crit == "pct-abs-obj":
        #     diff = (obj_full(alpha) - obj_full(alpha_minus1)).abs() / obj_full(alpha_minus1).abs()
        # elif crit == "abs-obj":
        # diff = (obj_full(alpha) - obj_full(alpha_minus1)).abs()
        diff = (obj_full(alpha) - obj_full(alpha_minus1)
                ).abs() / obj_full(alpha_minus1).abs()
        if verbose:
            print(diff)
        if diff.item() < tol:
            converged = True
            break
        alpha_minus2 = alpha_minus1.detach().clone()
        alpha_minus1 = alpha.detach().clone()
        epoch_restart += 1
    return alpha, monitored
    # if not converged:
    #     raise ConvergenceWarning("Maximum number of iteration reached")
