

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


def acc_proxgd(alpha0, prox, obj, grad, n_epoch=20000, tol=1e-6, beta=0.8, d=20, monitor=None):
    alpha_minus1 = alpha0
    alpha_minus2 = alpha0
    step_size = 1
    converged = False
    monitored = []
    for epoch in range(0, n_epoch):
        acc_cste = epoch / (epoch + 1 + d)
        alpha_v = alpha_minus1 + acc_cste * (alpha_minus1 - alpha_minus2)
        grad_v = grad(alpha_v)
        step_size = acc_proxgd_lsearch(
            prox, obj, step_size, alpha_v, grad_v, beta)
        alpha = prox(alpha_v - step_size * grad_v, step_size)
        if monitor is not None:
            monitored.append(monitor(alpha))
        diff = (alpha - alpha_minus1).norm() / alpha_minus1.norm()
        print(diff)
        if diff < tol:
            converged = True
            break
        alpha_minus2 = alpha_minus1.detach().clone()
        alpha_minus1 = alpha.detach().clone()
    return alpha, monitored


def acc_proxgd_restart(alpha0, prox, obj, grad, n_epoch=20000, tol=1e-6, beta=0.8, d=20, monitor=None):
    alpha_minus1 = alpha0
    alpha_minus2 = alpha0
    step_size = 1
    epoch_restart = 0
    converged = False
    monitored = []
    for epoch in range(0, n_epoch):
        acc_cste = epoch_restart / (epoch_restart + 1 + d)
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
        else:
            alpha = alpha_tentative
        if monitor is not None:
            monitored.append(monitor(alpha))
        diff = (alpha - alpha_minus1).norm() / alpha_minus1.norm()
        print(diff)
        if diff < tol:
            converged = True
            break
        alpha_minus2 = alpha_minus1.detach().clone()
        alpha_minus1 = alpha.detach().clone()
        epoch_restart += 1
    return alpha, monitored
    # if not converged:
    #     raise ConvergenceWarning("Maximum number of iteration reached")
