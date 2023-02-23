import numpy as np
import torch
from scipy.stats import *
import itertools
import scipy
from scipy.special import logsumexp

import sys


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


def is_pos_def_torch(x):
    test = torch.real(torch.linalg.eigvals(torch.real(x)))
    return torch.all(test > 0)


def make_pos_def_torch(x):
    """
    Force a square symmetric matrix to be positive semi definite
    """
    w, v = torch.linalg.eigh(torch.real(x))
    w_pos = torch.clip(w, 0, None)
    nonzero_w = w_pos[w_pos > 0]
    w_new = w_pos
    if nonzero_w.shape[0] > 0:
        if nonzero_w.shape[0] < w.shape[0]:
            # min_w = max(torch.max(nonzero_w)*1e-5,torch.min(nonzero_w))
            min_w = max(torch.max(nonzero_w) * 0.1, torch.min(nonzero_w))
            w_new = w_pos + min_w
    elif nonzero_w.shape[0] == 0:
        eprint(
            "No positive eigenvalues for A {}. w={} {}, w_pos={} {}, nonzero_w={}".format(
                x, w, w.shape, w_pos, w_pos.shape, nonzero_w
            )
        )
        w_new = torch.ones_like(w)
    x_star = v @ np.diag(w_new) @ v.T
    p = torch.sqrt(torch.sum(torch.abs(w)) / torch.sum(torch.abs(w_new)))
    return p * x_star


def make_pos_def(x):
    """
    Force a square symmetric matrix to be positive semi definite
    """
    w, v = np.linalg.eigh(x)
    w_pos = np.clip(w, 0, None)
    nonzero_w = w_pos[w_pos > 0]
    w_new = w_pos
    if nonzero_w.shape[0] > 0:
        if nonzero_w.shape[0] < w.shape[0]:
            min_w = max(np.max(nonzero_w) * 1e-5, np.min(nonzero_w))
            w_new = w_pos + min_w
    else:
        eprint(
            "No positive eigenvalues for A {}. w={} {}, w_pos={} {}, nonzero_w={}".format(
                x, w, w.shape, w_pos, w_pos.shape, nonzero_w
            )
        )
        w_new = np.ones_like(w)
    # w_neg = np.abs(np.clip(w,None,0))
    x_star = v @ np.diag(w_new) @ v.T
    p = np.sqrt(np.sum(np.abs(w)) / np.sum(w_new))
    return p * x_star


def safe_cholesky(A: torch.Tensor) -> torch.Tensor:
    eigenvalues, eigenvectors = torch.linalg.eigh(A)
    eigenvalues = torch.clamp(
        eigenvalues, min=1e-10
    )  # set any negative eigenvalues to a small positive value
    # L = eigenvectors @ torch.diag(eigenvalues.sqrt()) @ eigenvectors.t() #ZCA
    __, Rqr = torch.linalg.qr(torch.diag(eigenvalues.sqrt()) @ eigenvectors.t())
    Dg = torch.diag(torch.sign(torch.diag(Rqr)))
    Rqr = Dg @ Rqr
    return Rqr.T


# def make_pos_def(x):
#    sign, ld = np.linalg.slogdet(x)
#    while not np.isfinite(ld):
#        w,v = np.linalg.eigh(x)
#        maxw = np.max(np.abs(w))
#        w += 0.1 * maxw
#        print("log det of {} is not finite. Adding {} to eigvalues {} {}".format(x,maxw,w,v))
#        x = v @ np.diag(w) @ v.T
#        print("New x is {}".format(x))
#        sign, ld = np.linalg.slogdet(x)
#        #sys.exit(0)
#    return x
#
def safe_logdet(x):
    sign, ld = np.linalg.slogdet(x)
    while not np.isfinite(ld):
        w, v = np.linalg.eigh(x)
        maxw = np.max(np.abs(w))
        w += 0.1 * maxw
        print(
            "log det of {} is not finite. Adding {} to eigvalues {} {}".format(
                x, maxw, w, v
            )
        )
        x = v @ np.diag(w) @ v.T
        print("New x is {}".format(x))
        sign, ld = np.linalg.slogdet(x)
        # sys.exit(0)
    return sign, ld


def safe_inv(x):
    try:
        return np.linalg.pinv(x)
    except Exception as ex:
        print(ex)
        # print(x)
        sys.exit(0)
        x = make_pos_def(x)
        print(x)
        sys.exit(0)

    w, v = np.linalg.eigh(x)
    if np.any(w < 0):
        print("WARNING: negative eigenvalue in inverse of ", x, w, v)
        # w[w<0] = 0
        w = np.abs(w)
    if np.any(w == 0):
        print("WARNING: zero eigenvalue in inverse of ", x, w, v)
        maxw = np.max(np.abs(w))
        minw = np.min(np.abs(w[w != 0]))
        # w[w==0] = minw
        # w += minw
        w += 0.001 * minw
        print("new eigenvalues", w)
    return v @ np.diag(w ** (-1)) @ v.T


def sum_along_axis(a, axis=1):
    # assumes it is a numpy array
    if a.ndim <= axis:
        return a
    else:
        return a.sum(axis=axis)


# def logsumexp(log_values,**kwargs):
#    """
#    Prevents overflow in summing vectors of large numbers.
#    Parameters
#    ----------
#        ns: numpy array of natural logarithm transformed values to sum
#    Returns
#    -------
#        single value or numpy array of the natural logarithm-transformed summed value(s)
#    """
#    if log_values.ndim > 1:
#        #m = np.max(log_values,keepdims=True,axis=kwargs.get('axis'))
#        m = np.max(log_values,**kwargs)
#        #if np.isneginf(m).any():
#        #    return np.NINF
#        sumOfExp = np.exp(log_values - m[:,None]).sum(**kwargs)
#        return m + np.log(sumOfExp)
#    else:
#        m = np.max(log_values,**kwargs)
#        sumOfExp = np.exp(log_values - m).sum(**kwargs)
#        return m + np.log(sumOfExp)


def gen_dict_extract(key, var):
    if hasattr(var, "iteritems"):
        for k, v in var.iteritems():
            if k == key:
                yield v
            if isinstance(v, dict):
                for result in gen_dict_extract(key, v):
                    yield result
            elif isinstance(v, list):
                for d in v:
                    for result in gen_dict_extract(key, d):
                        yield result
