import jax.numpy as np
import skbio.stats.composition as cmp
import jax

import os
import sys
sys.path.append(os.getcwd())
from method_fct import LinearInstrumentModel


# ----------------------------------------------------------------------------------------------------------------------
# DATA SIMULATION - Negative Binomial Model
# ----------------------------------------------------------------------------------------------------------------------
def sim_FS_negbinomial(key,
                      n: int,
                      p: int,
                      num_inst: int,
                      mu_c: int,
                      alpha0: np.ndarray,
                      alphaT: np.ndarray,
                      c_X: int,
                     ps: float=None,
                     Z_max: float=10):

    """Simulate first stage as a negative binomial function

    Parameters
    ----------
    key : jax PRNG key
        random seed
    n : int
        number of generated samples
    p : int
        number of generated microbes
    num_inst: int
        number of generated instruments
    mu_c: np.ndarray
        compositional vector which is used in the confounding term
    alpha0: np.ndarray
        intercept for the relationship between Z and X
    alphaT: np.ndarray
        slope for the relationship between Z and X
    c_X: int
        dispersion parameter for the distribution of X
    ps: np.ndarray=None
        zero inflation parameter
    Z_max: float=10
        upper bound for sampled Z

    Returns
    -------
    Z_sim : np.ndarray
        matrix of simulated Z values
    X_power : np.ndarray
        unconfounded X values
    X_sim : np.ndarray
        matrix of simulated X values
    confounder_perturb : np.ndarray
        matrix of confounder values
    """

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


def sim_IV_negbinomial(key,
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
    """Simulate entire setup of IV with first stage as a negative binomial function and second stage as linear

        Parameters
        ----------
        key : jax PRNG key
            random seed
        n : int
            number of generated samples
        p : int
            number of generated microbes
        num_inst: int
            number of generated instruments
        mu_c: np.ndarray
            compositional vector which is used in the confounding term
        alpha0: np.ndarray
            intercept for the relationship between Z and X
        alphaT: np.ndarray
            slope for the relationship between Z and X
        c_X: int
            dispersion parameter for the distribution of X
        c_Y : int
            strength of confounding on Y
        beta0 : int
            intercept of true causal beta parameter
        betaT : np.ndarray
            slope of true causal beta parameter
        ps: np.ndarray=None
            zero inflation parameter
        Z_max: float=10
            upper bound for sampled Z

        Returns
        -------
        confounder : np.ndarray
            matrix of confounder values
        Z_sim : np.ndarray
            matrix of simulated Z values
        X_sim : np.ndarray
            matrix of simulated X values
        Y_sim : np.ndarray
            matrix of simulated Y values
        X_star : np.ndarray
            matrix of unconfounded X values, generated for evaluation of true causal effect
        Y_star : np.ndarray
            matrix of unconfounded Y values, computed from X_star values and indicating the true causal effect of X on Y
        """

    Z_sim, X_sim, X_sim_ilr, confounder = sim_FS_negbinomial(key, n, p, num_inst, mu_c, alpha0, alphaT, c_X, ps, Z_max)
    # derive Y variable
    confounder_n0 = LinearInstrumentModel.add_term_to_zeros(confounder)
    X_sim = LinearInstrumentModel.add_term_to_zeros(X_sim)
    Y_sim = beta0 + np.log(X_sim) @ betaT + np.log(confounder_n0) @ c_Y

    # compute intervention set up
    key, subkey = jax.random.split(key)
    _, X_star, X_star_ilr, _ = sim_FS_negbinomial(key, num_star, p, num_inst, mu_c, alpha0, alphaT, c_X, ps, Z_max)
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


# ----------------------------------------------------------------------------------------------------------------------
# DATA SIMULATION - ILR coordinate simulation
# ----------------------------------------------------------------------------------------------------------------------
def sim_FS_ilr_linear(key,
                      n: int,
                      p: int,
                      num_inst: int,
                      mu_c: int,
                      alpha0: np.ndarray,
                      alphaT: np.ndarray,
                      c_X: np.ndarray):
    """Simulate linear first stage of IV approach on ilr coordinates

    Parameters
    ----------
    key : jax PRNG key
        random seed
    n : int
        number of generated samples
    p : int
        number of generated microbes
    num_inst: int
        number of generated instruments
    mu_c: int
        offset for confounding variable (mean of confounder)
    alpha0: np.ndarray
        intercept for the relationship between Z and X
    alphaT: np.ndarray
        slope for the relationship between Z and X
    c_X: int
        spreading of confounding variable (std of confounder)


    Returns
    -------
    Z_sim : np.ndarray
        matrix of simulated Z values
    X_sim : np.ndarray
        matrix of simulated X values
    X_sim_ilr : np.ndarray
        ilr coordinates, matrix of simulated X values
    confounder : np.ndarray
        matrix of confounder values
    """
    key, subkey = jax.random.split(key)
    Z_sim = jax.random.uniform(subkey, (n, num_inst))

    # initiate confounder
    key, subkey = jax.random.split(key)
    confounder = mu_c + jax.random.normal(subkey, (n, 1))

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

    """Simulate entire setup of IV with first and second stage as linear in ilr coordinates

        Parameters
        ----------
        key : jax PRNG key
            random seed
        n : int
            number of generated samples
        p : int
            number of generated microbes
        num_inst: int
            number of generated instruments
        mu_c: int
            offset for confounding variable (mean of confounder)
        alpha0: np.ndarray
            intercept for the relationship between Z and X
        alphaT: np.ndarray
            slope for the relationship between Z and X
        c_X: int
            spreading of confounding variable (std of confounder)
        c_Y : int
            strength of confounding on Y
        beta0 : int
            intercept of true causal beta parameter
        betaT : np.ndarray
            slope of true causal beta parameter
        num_star : int
            number of evaluated ground truth values for true causal effect


        Returns
        -------
        confounder : np.ndarray
            matrix of confounder values
        Z_sim : np.ndarray
            matrix of simulated Z values
        X_sim : np.ndarray
            matrix of simulated X values
        Y_sim : np.ndarray
            matrix of simulated Y values
        X_star : np.ndarray
            matrix of unconfounded X values, generated for evaluation of true causal effect
        Y_star : np.ndarray
            matrix of unconfounded Y values, computed from X_star values and indicating the true causal effect of X on Y
        """
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

    """Simulate non-linear first stage of IV approach on ilr coordinates

    Parameters
    ----------
    key : jax PRNG key
        random seed
    n : int
        number of generated samples
    p : int
        number of generated microbes
    num_inst: int
        number of generated instruments
    mu_c: int
        offset for confounding variable (mean of confounder)
    alpha0: np.ndarray
        intercept for the relationship between Z and X
    alphaT: np.ndarray
        slope for the relationship between Z and X
    c_X: int
        spreading of confounding variable (std of confounder)


    Returns
    -------
    Z_sim : np.ndarray
        matrix of simulated Z values
    X_sim : np.ndarray
        matrix of simulated X values
    X_sim_ilr : np.ndarray
        ilr coordinates, matrix of simulated X values
    confounder : np.ndarray
        matrix of confounder values
    """

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





