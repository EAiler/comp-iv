import jax.numpy as np
import jax
import numpy as onp
import matplotlib.pylab as plt
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from statsmodels.sandbox.regression.gmm import IV2SLS
import statsmodels.api as sm
import skbio.stats.composition as cmp
import jax
import dirichlet
import time
import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "generalIVmodel/"))
sys.path.append(os.path.join(os.getcwd(), "generalIVmodel/src"))
from method_fct import LinearInstrumentModel


def sim_FS_ilr_normalZ(key,
                      n: int,
                      p: int,
                      num_inst: int,
                      mu_c: int,
                      alpha0: np.ndarray,
                      alphaT: np.ndarray,
                      c_X: np.ndarray):
    """Simulate the microbiome by its ilr components and being linear in its dependence of Z"""
    key, subkey = jax.random.split(key)
    Z_sim = jax.random.normal(subkey, (n, num_inst))

    # initiate confounder
    key, subkey = jax.random.split(key)
    confounder = mu_c + jax.random.normal(subkey, (n, 1))

    # derive X variable

    # derive X variable
    Z_influence = np.array([Z_sim @ alphaT[j, :] for j in range(p - 1)]).T
    X_sim_ilr = np.stack([
        alpha0[i] + Z_influence[:, i] + c_X[i] * confounder.squeeze() for i in range(p - 1)]).T

    X_sim = cmp.ilr_inv(X_sim_ilr)

    return Z_sim, X_sim, X_sim_ilr, confounder


def sim_IV_ilr_linear_normalZ(key,
                  n: int,
                  p: int,
                  num_inst: int,
                  mu_c: int,
                  c_X: np.ndarray,
                  alpha0: np.ndarray,
                  alphaT: np.ndarray,
                  c_Y: int,
                  beta0: float,
                  betaT: np.ndarray,
                num_star:int = 500):

    """Simulate IV setup with ilr linearity in all components """
    Z_sim, X_sim, X_sim_ilr, confounder = sim_FS_ilr_normalZ(key, n, p, num_inst, mu_c, alpha0, alphaT, c_X)
    # derive Y variable
    Y_sim = beta0 + X_sim_ilr @ betaT + c_Y * confounder.squeeze()

    # compute intervention set up
    key, subkey = jax.random.split(key)
    _, X_star, X_star_ilr, _ = sim_FS_ilr_normalZ(key, num_star, p, num_inst, mu_c, alpha0, alphaT, c_X)

    Y_star = beta0 + X_star_ilr @ betaT + c_Y * mu_c

    return confounder, Z_sim, X_sim, Y_sim, X_star, Y_star


def sim_FS_dirichlet(key,
                      n: int,
                      p: int,
                      num_inst: int,
                      mu_c: int,
                      alpha0: np.ndarray,
                      alphaT: np.ndarray,
                      c_X: int):
    """Simulate the microbiome from a dirichlet distribution """
    key, subkey = jax.random.split(key)
    Z_sim = jax.random.uniform(subkey, (n, num_inst))

    # initiate confounder
    key, subkey = jax.random.split(key)
    confounder = mu_c + jax.random.uniform(subkey, (n, p))
    # derive X variable
    Z_influence = np.array([Z_sim @ alphaT[j, :] for j in range(p)]).T
    alpha = np.stack([alpha0[i] + Z_influence[:, i] for i in range(p)]).T
    alpha = alpha.clip(0)  # clip the 0 -> not allowed as parameters
    X_sim = jax.random.dirichlet(key, alpha, (n,))
    X_perturb = jax.random.dirichlet(subkey, confounder, (n, ))
    X_sim = cmp.perturb(X_sim, cmp.power(X_perturb, c_X))

    X_n0 = LinearInstrumentModel.add_term_to_zeros(X_sim, add_term=0.001)
    X_perturb = LinearInstrumentModel.add_term_to_zeros(X_perturb, add_term=0.001)
    X_sim_ilr = cmp.ilr(X_n0)

    return Z_sim, X_sim, X_sim_ilr, confounder, X_perturb


def sim_FS_lognormal(key,
                      n: int,
                      p: int,
                      num_inst: int,
                      mu_c: int,
                      alpha0: np.ndarray,
                      alphaT: np.ndarray,
                      c_X: int,
                     ps: float=None,
                     Z_max: float=10):
    Sigma = np.identity(p)
    dispersion = c_X
    key, subkey = jax.random.split(key)
    confounder = jax.random.uniform(subkey, (n,), minval=0.2, maxval=3)

    # error
    key, subkey1, subkey2 = jax.random.split(key, 3)

    # simulate Z
    key, subkey = jax.random.split(key)
    Z_sim = jax.random.uniform(subkey, (n, num_inst), minval=1, maxval=Z_max)
    # simulate x
    Z_influence = Z_sim@alphaT
    mu = np.array([alpha0[iter] + Z_influence[:, iter] for iter in range(p)]).T
    mu = mu.clip(0)  # clip the 0 -> not allowed as parameters
    # TODO : reinstate that
    #log_mu = np.array([alpha0[iter] + (Z_sim * alphaT[:, iter]).sum(axis=1) for iter in range(p)]).T
    #mu = np.exp(log_mu)  # clip the 0 -> not allowed as parameters
    from helper_fct import multi_negative_binomial
    X_sim = multi_negative_binomial(p, mu, Sigma, dispersion, n, ps=ps)

    X_power = np.empty((p,))
    confounder_perturb = np.empty((p, ))
    for i in range(n):
        con = cmp.power(mu_c, confounder[i])
        X_power = np.vstack([X_power, cmp.perturb(X_sim[i, :], con)])
        confounder_perturb = np.vstack([confounder_perturb, con])

    X_sim_ilr = cmp.ilr(X_power[1:, :])

    return Z_sim, X_power[1:, :], X_sim_ilr, confounder_perturb[1:, :]




def sim_IV_lognormal_linear(key,
                  n: int,
                  p: int,
                  num_inst: int,
                  mu_c: int,
                  c_X: int,
                  alpha0: np.ndarray,
                  alphaT: np.ndarray,
                  c_Y: np.ndarray,
                  beta0: float,
                  betaT: np.ndarray,
                num_star:int = 500,
                            ps: float = None,
                            Z_max: float=10):

    """Simulate IV setup with ilr linearity in all components """
    Z_sim, X_sim, X_sim_ilr, confounder = sim_FS_lognormal(key, n, p, num_inst, mu_c, alpha0, alphaT, c_X, ps, Z_max)
    # derive Y variable
    confounder_n0 = LinearInstrumentModel.add_term_to_zeros(confounder)
    X_sim = LinearInstrumentModel.add_term_to_zeros(X_sim)
    Y_sim = beta0 + np.log(X_sim) @ betaT + np.log(confounder_n0) @ c_Y

    # compute intervention set up
    key, subkey = jax.random.split(key)
    _, X_star, X_star_ilr, _ = sim_FS_lognormal(key, num_star, p, num_inst, mu_c, alpha0, alphaT, c_X, ps, Z_max)
    X_star = LinearInstrumentModel.add_term_to_zeros(X_star)
    Y_star = np.array([])
    num = 80
    for _ in range(num):
        key, subkey = jax.random.split(key, )
        confounder_interim = jax.random.uniform(subkey, (num_star,), minval=0.2, maxval=3)
        X_perturb = np.array([cmp.power(mu_c, confounder_interim[i]) for i in range(num_star)])
        X_perturb = LinearInstrumentModel.add_term_to_zeros(X_perturb, add_term=0.001)
        Y_star_step = beta0 + np.log(X_star) @ betaT + np.log(X_perturb) @ c_Y
        Y_star = np.append(Y_star, Y_star_step)
    Y_star = Y_star.reshape((num, num_star)).mean(axis=0)

    return confounder, Z_sim, X_sim, Y_sim, X_star, Y_star



def sim_IV_dirichlet_linear(key,
                  n: int,
                  p: int,
                  num_inst: int,
                  mu_c: int,
                  c_X: int,
                  alpha0: np.ndarray,
                  alphaT: np.ndarray,
                  c_Y: int,
                  beta0: float,
                  betaT: np.ndarray,
                num_star:int = 500):

    """Simulate IV setup with ilr linearity in all components """
    Z_sim, X_sim, X_sim_ilr, confounder, X_perturb = sim_FS_dirichlet(key, n, p, num_inst, mu_c, alpha0, alphaT, c_X)
    # derive Y variable
    Y_sim = beta0 + X_sim_ilr @ betaT + c_Y * np.log(X_perturb)@np.array([2, 2, -4])
    # compute intervention set up
    key, subkey = jax.random.split(key)
    _, X_star, X_star_ilr, _, _ = sim_FS_dirichlet(key, num_star, p, num_inst, mu_c, alpha0, alphaT, c_X)

    # TODO : note that we replaced the confounder simulation by a uniform variable!
    print("Start Iteration")
    Y_star = np.array([])
    num = 80
    for _ in range(num):
        key, subkey = jax.random.split(key, )
        confounder_interim = mu_c + jax.random.uniform(key, (num_star, p))
        X_perturb = jax.random.dirichlet(subkey, confounder_interim, (num_star,))
        X_perturb = LinearInstrumentModel.add_term_to_zeros(X_perturb, add_term=0.001)
        Y_star_step = beta0 + X_star_ilr @ betaT + c_Y * np.log(X_perturb)@np.array([2, 2, -4])
        Y_star = np.append(Y_star, Y_star_step)
    Y_star = Y_star.reshape((num, num_star)).mean(axis=0)

    return confounder, Z_sim, X_sim, Y_sim, X_star, Y_star


def sim_FS_ilr_linear(key,
                      n: int,
                      p: int,
                      num_inst: int,
                      mu_c: int,
                      alpha0: np.ndarray,
                      alphaT: np.ndarray,
                      c_X: np.ndarray):
    """Simulate the microbiome by its ilr components and being linear in its dependence of Z"""
    key, subkey = jax.random.split(key)
    Z_sim = jax.random.uniform(subkey, (n, num_inst))

    # initiate confounder
    key, subkey = jax.random.split(key)
    confounder = mu_c + jax.random.normal(subkey, (n, 1))

    # derive X variable

    # derive X variable
    Z_influence = np.array([Z_sim @ alphaT[j, :] for j in range(p - 1)]).T
    X_sim_ilr = np.stack([
        alpha0[i] + Z_influence[:, i] + c_X[i] * confounder.squeeze() for i in range(p - 1)]).T

    X_sim = cmp.ilr_inv(X_sim_ilr)

    return Z_sim, X_sim, X_sim_ilr, confounder

def sim_IV_ilr_linear(key,
                  n: int,
                  p: int,
                  num_inst: int,
                  mu_c: int,
                  c_X: np.ndarray,
                  alpha0: np.ndarray,
                  alphaT: np.ndarray,
                  c_Y: int,
                  beta0: float,
                  betaT: np.ndarray,
                num_star:int = 500):

    """Simulate IV setup with ilr linearity in all components """
    Z_sim, X_sim, X_sim_ilr, confounder = sim_FS_ilr_linear(key, n, p, num_inst, mu_c, alpha0, alphaT, c_X)
    # derive Y variable
    Y_sim = beta0 + X_sim_ilr @ betaT + c_Y * confounder.squeeze()

    # compute intervention set up
    key, subkey = jax.random.split(key)
    _, X_star, X_star_ilr, _ = sim_FS_ilr_linear(key, num_star, p, num_inst, mu_c, alpha0, alphaT, c_X)

    Y_star = beta0 + X_star_ilr @ betaT + c_Y * mu_c

    return confounder, Z_sim, X_sim, Y_sim, X_star, Y_star


def sim_IV_ilr_nonlinear(key,
                      n: int,
                      p: int,
                      num_inst: int,
                      mu_c: int,
                      c_X: np.ndarray,
                      alpha0: np.ndarray,
                      alphaT: np.ndarray,
                      c_Y: int,
                      beta0: float,
                      betaT: np.ndarray,
                      num_star=500):

    """Simulate IV setup with ilr linearity in the first stage, but a nonlinear, misspecified second stage """
    Z_sim, X_sim, X_sim_ilr, confounder = sim_FS_ilr_linear(key, n, p, num_inst,
                                                            mu_c, alpha0, alphaT, c_X)
    # derive Y variable
    Y_sim = beta0 + 1/10*(X_sim_ilr @ betaT) + 1/20*((X_sim_ilr+1) ** 3 @ np.ones(p - 1, )) + c_Y * confounder.squeeze()

    # compute intervention set up
    key, subkey = jax.random.split(key)
    _, X_star, X_star_ilr, _ = sim_FS_ilr_linear(key, num_star, p, num_inst,
                                                 mu_c, alpha0, alphaT, c_X)

    Y_star = beta0 + 1/10*(X_star_ilr @ betaT) + 1/20*((X_star_ilr+1) ** 3 @ np.ones(p - 1, )) + c_Y * mu_c

    return confounder, Z_sim, X_sim, Y_sim, X_star, Y_star





