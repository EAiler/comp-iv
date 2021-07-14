
from jax import numpy as np
from rpy2.robjects.packages import importr
from rpy2.robjects import FloatVector, IntVector
from rpy2.robjects import r
importr('SpiecEasi')
importr('Compositional')
importr('vegan')
import random as rm


def multi_negative_binomial(p, mu, Sigma, dispersion, num_samples, ps=None):
    """simulate from a negative binomial - count data sampling
    Parameter
    --------
    p: int
        number of microbiota dimensions
    mu: numpy array
    dispersion: int
        dispersion parameter
    num_samples: int
        number of samples to generate, should correspond to dimensions of mu
    ps: float
        probability of zero inflation

    Returns
    _______
    X_sim: numpy matrix
        matrix containing percentage points
    """
    XsimNB = np.empty((0, p))
    Sigma_r = r.matrix(IntVector(Sigma.reshape(-1)), nrow=int(p))

    # for each mu create an individual sample
    for i in range(num_samples):
        mu_r = FloatVector(mu[i, :])
        xsimNB_r = r["rmvnegbin"](2, mu_r, Sigma_r, dispersion)
        xsimNB = np.asarray(xsimNB_r)[1, :]

        while np.count_nonzero(np.isnan(xsimNB)) != 0:
            xsimNB_r = r["rmvnegbin"](2, mu_r, Sigma_r, dispersion)
            xsimNB = np.asarray(xsimNB_r)[1, :]

        XsimNB = np.vstack([XsimNB, xsimNB])
    # integrate zero inflation manually
    if ps is not None:
        mat = np.empty((0, num_samples))
        for i in range(p):
            sim = np.array(rm.choices([0, 1], weights=[ps[i], 1-ps[i]], k=num_samples))
            mat = np.vstack([mat, sim])
        XsimNB = mat.T * XsimNB
    X_sim = np.apply_along_axis(lambda x: x / x.sum(), 1, XsimNB)

    return X_sim


def whiten_data(col, verbose=False):
    """whiten the data, if verbose set to true also returns the mean and std

    Parameters
    ----------
    col : np.ndarray
        matrix of data that should be whitened
    verbose : bool=False
        bool if set to true will output also the mean and std of the whitened data

    Returns
    -------
    X : np.ndarray
        whitened data

    if verbose == True
    mu_c : float
        mean of whitened data
    std_c : float
        std of whitened data
    X : np.ndarray
        whitened data

    """
    mu_c = col.mean()
    std_c = col.std()
    if verbose:
        return mu_c, std_c, (col - col.mean()) / col.std()
    else:
        return (col - col.mean()) / col.std()


def diversity(X, method="shannon"):
    """Computation of diversity, X has rows with the samples and the columns are the bacteria

    Parameters
    ----------
    X : np.ndarray
        matrix with microbiom data, the rows are the different samples and the columns hold the different bacteria
    method : {"shannon", "simpson"}
        string for the different diversity methods, computation is done via the R package vegan

    """
    diversity_vec = []
    for i in range(X.shape[0]):
        vec_r = FloatVector(X[i, :])
        div = r["diversity"](vec_r, index=method)
        div = np.asarray(div)
        diversity_vec.append(div)

    return np.array(diversity_vec)




