# standard libraries
import logging
import jax
import jax.numpy as np
import numpy as onp
import skbio.stats.composition as cmp
from simulate_data_fct import sim_IV_ilr_linear, sim_IV_ilr_nonlinear, sim_IV_negbinomial
from types import SimpleNamespace

def return_parameter_setting(strDim, strRel, strInst,
                             alpha0_hat=None, alphaT_hat=None, ps_hat=None, betahat=None):
    """Returns the parameter settings of the paper

        Parameters
        ----------
        strDim : {p3, p20}
            string including the dimensionality
        strRel : {linear, nonlinear, negBinom}
            string including the relationship in the first/second stage
        strInst : {weak, medium, strong, unknown}
            string including the strength of the instrument

        Returns
        -------
        dict : dict
            returns a dictionary including all the relevant parameters
        """

    key = jax.random.PRNGKey(253)
    ps = None  # zero inflation parameter, only necessary for negBinom option

    # ------------------------------------------------------------------------------------------------------------------
    # p3, linear, medium
    # ------------------------------------------------------------------------------------------------------------------
    if (strDim == "p3") & (strRel == "linear") & (strInst == "medium"):
        n = 1000  # number of samples
        p = 3  # number of microbes
        V = cmp._gram_schmidt_basis(p)
        num_inst = 2  # number of instruments
        c_X = np.array([0.5, 0.5])  # confounder multiplication to X
        alpha0 = np.array([1, 1])  # INSTRUMENT intercept
        alphaT = np.array([[0.5, -0.15],  # INSTRUMENT INTERCEPT First Component of Microbiome
                           [0.3, 0.7]])  # INSTRUMENT INTERCEPT Second Component of Microbiome

        beta0 = 5  # true intercept of causal relationship in ilr coordinates
        betaT = np.array([4, 1])  # true slope of causal relationship in ilr coordinates
        mu_c = -3  #
        c_Y = 4  #

        betaT_log = V.T @ betaT  # conversion of ilr coordinate parameters to


    # ------------------------------------------------------------------------------------------------------------------
    # p3, linear, weak
    # ------------------------------------------------------------------------------------------------------------------
    elif (strDim == "p3") & (strRel == "linear") & (strInst == "weak"):
        n = 1000
        p = 3
        V = cmp._gram_schmidt_basis(p)
        num_inst = 2
        c_X = np.array([1, 1])  # confounder multiplication to X
        alpha0 = np.array([4, 1])  # INSTRUMENT intercept
        alphaT = np.array([[+0.15, 0.15],  # INSTRUMENT INTERCEPT FIrst Component of Microbiome
                           [0.2, 0]])  # INSTRUMENT INTERCEPT Second Component of Microbiome
        beta0 = 2
        betaT = np.array([6, 2])
        betaT_log = V.T @ betaT
        mu_c = -2
        c_Y = 4


    # ------------------------------------------------------------------------------------------------------------------
    # p3, nonlinear, strong
    # ------------------------------------------------------------------------------------------------------------------
    elif (strDim == "p3") & (strRel == "nonlinear") & (strInst == "strong"):
        n = 1000
        p = 3
        V = cmp._gram_schmidt_basis(p)

        num_inst = 2
        c_X = np.array([2, 2])  # confounder multiplication to X
        alpha0 = np.array([1, 1])  # INSTRUMENT intercept
        alphaT = np.array([[4, 1],  # INSTRUMENT INTERCEPT FIrst Component of Microbiome
                           [-1, 3]])  # INSTRUMENT INTERCEPT Second Component of Microbiome
        beta0 = 5
        betaT = np.array([6, 2])
        mu_c = -1
        c_Y = 4
        betaT_log = None


    # ------------------------------------------------------------------------------------------------------------------
    # p20, linear, unkown
    # ------------------------------------------------------------------------------------------------------------------
    elif (strDim == "p20") & (strRel == "linear") & (strInst == "unknown"):
        n = 1000
        p = 30
        num_inst = 20
        V = cmp._gram_schmidt_basis(p)
        c_X = np.hstack([np.array([-1, 2, -2, 1]), np.zeros(p - 1 - 4, )])  # confounder multiplication to X
        alpha0 = np.hstack([np.array([1, 1, 3, 1]), np.zeros(p - 1 - 4, )])
        alphaT = jax.random.choice(key, np.array([0.5, 0.75, 0.25, 0, 0, 0]), (p - 1, num_inst))  # -4, 4
        betaT_log = np.hstack([np.array([5, -5, 3, -3]), np.zeros(p - 1 - 4, )])
        betaT_p = np.hstack([betaT_log, -betaT_log.sum()])
        beta0 = 5
        betaT = V @ betaT_p

        mu_c = 3
        c_Y = 4


    # ------------------------------------------------------------------------------------------------------------------
    # p20, negative Binomial, unkown
    # ------------------------------------------------------------------------------------------------------------------
    elif (strDim == "p20") & (strRel == "negbinom") & (strInst == "unknown"):
        key = jax.random.PRNGKey(191)
        n = 500
        p = 30
        V = cmp._gram_schmidt_basis(p)
        num_inst = 20
        c_X = 2  # dispersion parameter

        # instrument strength
        alpha0 = np.hstack(
            [np.array([1, 1, 2, 1, 4, 4, 2, 1, 4, 4, 2, 1]), jax.random.choice(key, np.array([1, 2, 2]), (p - 8,))])
        alphaT = jax.random.choice(key, np.array([0, 0, 0, 10]), (num_inst, p))
        # confounder
        mu_c = np.hstack([np.array([0.2, 0.3, 0.2, 0.1]), jax.random.uniform(key, (p - 4,), minval=0.01, maxval=0.05)])
        mu_c = mu_c / mu_c.sum()  # has to be a compositional vector
        ps = np.hstack([np.zeros((int(p / 2),)), 0.8 * np.ones((p - int(p / 2),))])  # prob of zero inflation

        # relationship between X and Y
        beta0 = 1
        betaT = np.hstack([np.array([-5, -5, -5, -5]), np.array([5, 5, 5, 5]),
                           np.zeros((p - 8))])  # beta is chosen to sum up to one
        betaT_log = betaT
        # confounder influence to Y
        c_Y = np.hstack([np.array([20, 20, 0, 0]), np.array([-5, -5, -5, -5, -5, -5, -5, -5]),
                         np.zeros((p - 12))])  # confounder is a composition as well.


    elif (strDim == "p47") & (strRel == "negbinom") & (strInst == "unknown"):
        n = 1000
        p = 47

        V = cmp._gram_schmidt_basis(p)
        num_inst = 1
        c_X = 0.2

        # instrument strength
        alpha0 = alpha0_hat
        alphaT = alphaT_hat[..., np.newaxis].T

        mu_c = onp.zeros((p,))
        mu_c[np.where(alpha0 > 100)] = jax.random.uniform(key, ((len(np.where(alpha0 > 100)[0])),), minval=3, maxval=10)
        mu_c = mu_c / mu_c.sum()  # has to be a compositional vector

        ps = ps_hat
        # relationship between X and Y
        beta0 = float(betahat[0])
        betaT = betahat[1:]  # beta is chosen to sum up to one
        # confounder influence to Y
        c_Y_helper = onp.zeros((p,))
        c_Y_helper[np.where(alpha0 > 100)] = jax.random.choice(key, np.array([-3, 3]),
                                                               ((len(np.where(alpha0 > 100)[0])),))
        c_Y = np.hstack([c_Y_helper[:-1], -c_Y_helper[:-1].sum()])
        betaT_log = betaT

    else:
        print("This combination is not implemented....")

    dictReturn = {
        "strDim": strDim,
        "strRel": strRel,
        "strInst": strInst,
        "n": n,
        "p": p,
        "num_inst": num_inst,
        "c_X": c_X,
        "alpha0": alpha0,
        "alphaT": alphaT,
        "beta0": beta0,
        "betaT": betaT,
        "mu_c": mu_c,
        "c_Y": c_Y,
        "betaT_log": betaT_log,
        "ps" : ps,
        "V" : V,
        "key": key
    }
    return dictReturn


def simulate_data(dictReturn):
    """Returns the parameter settings of the paper as well as a first simulation of the data

    Parameters
    ----------
    strDim : {p3, p20}
        string including the dimensionality
    strRel : {linear, nonlinear, negBinom}
        string including the relationship in the first/second stage
    strInst : {weak, medium, strong, unknown}
        string including the strength of the instrument

    Returns
    -------
    confounder : np.ndarray
        sample matrix of confounder
    Z_sim : np.ndarray
        sample matrix of instrument
    X_sim : np.ndarray
        sample matrix of microbiome data, compositional data
    Y_sim : np.ndarray
        sample matrix of output
    X_star : np.ndarray
        sample matrix of interventional microbiome data, compositional data
    Y_star : np.ndarray
        sample matrix of true causal effect based on the interventional data X_star
    """

    param = SimpleNamespace(**dictReturn)
    # ------------------------------------------------------------------------------------------------------------------
    # p3, linear, medium
    # ------------------------------------------------------------------------------------------------------------------
    if (param.strDim == "p3") & (param.strRel == "linear") & (param.strInst == "medium"):
        # simulation of data
        confounder, Z_sim, X_sim, Y_sim, X_star, Y_star = sim_IV_ilr_linear(
            param.key, n=param.n, p=param.p,
            num_inst=param.num_inst, mu_c=param.mu_c, c_X=param.c_X, alpha0=param.alpha0, alphaT=param.alphaT,
            c_Y=param.c_Y, beta0=param.beta0, betaT=param.betaT, num_star=500)
        return confounder, Z_sim, X_sim, Y_sim, X_star, Y_star

    # ------------------------------------------------------------------------------------------------------------------
    # p3, linear, weak
    # ------------------------------------------------------------------------------------------------------------------
    elif (param.strDim == "p3") & (param.strRel == "linear") & (param.strInst == "weak"):
        confounder, Z_sim, X_sim, Y_sim, X_star, Y_star = sim_IV_ilr_linear(
            param.key, n=param.n, p=param.p,
            num_inst=param.num_inst, mu_c=param.mu_c, c_X=param.c_X, alpha0=param.alpha0, alphaT=param.alphaT,
            c_Y=param.c_Y, beta0=param.beta0, betaT=param.betaT, num_star=500)
        return confounder, Z_sim, X_sim, Y_sim, X_star, Y_star

    # ------------------------------------------------------------------------------------------------------------------
    # p3, nonlinear, strong
    # ------------------------------------------------------------------------------------------------------------------
    elif (param.strDim == "p3") & (param.strRel == "nonlinear") & (param.strInst == "strong"):

        confounder, Z_sim, X_sim, Y_sim, X_star, Y_star = sim_IV_ilr_nonlinear(
            param.key, n=param.n, p=param.p,
            num_inst=param.num_inst, mu_c=param.mu_c, c_X=param.c_X, alpha0=param.alpha0, alphaT=param.alphaT, c_Y=param.c_Y,
            beta0=param.beta0, betaT=param.betaT, num_star=500)
        return confounder, Z_sim, X_sim, Y_sim, X_star, Y_star
    # ------------------------------------------------------------------------------------------------------------------
    # p20, linear, unkown
    # ------------------------------------------------------------------------------------------------------------------
    elif (param.strDim == "p20") & (param.strRel == "linear") & (param.strInst == "unknown"):

        confounder, Z_sim, X_sim, Y_sim, X_star, Y_star = sim_IV_ilr_nonlinear(
            param.key, n=param.n, p=param.p,
            num_inst=param.num_inst, mu_c=param.mu_c, c_X=param.c_X,
            alpha0=param.alpha0, alphaT=param.alphaT, c_Y=param.c_Y,
            beta0=param.beta0, betaT=param.betaT, num_star=500)

        return confounder, Z_sim, X_sim, Y_sim, X_star, Y_star
    # ------------------------------------------------------------------------------------------------------------------
    # p20, negative Binomial, unkown
    # ------------------------------------------------------------------------------------------------------------------
    elif (param.strDim == "p20") & (param.strRel == "negbinom") & (param.strInst == "unknown"):

        confounder, Z_sim, X_sim, Y_sim, X_star, Y_star = sim_IV_negbinomial(
            param.key, n=param.n, p=param.p,
            num_inst=param.num_inst, mu_c=param.mu_c, c_X=param.c_X,
            alpha0=param.alpha0, alphaT=param.alphaT, c_Y=param.c_Y,
            beta0=param.beta0, betaT=param.betaT, num_star=250, ps=param.ps)

        return confounder, Z_sim, X_sim, Y_sim, X_star, Y_star

    # ------------------------------------------------------------------------------------------------------------------
    # p47, negative Binomial, unkown
    # ------------------------------------------------------------------------------------------------------------------
    elif (param.strDim == "p47") & (param.strRel == "negbinom") & (param.strInst == "unknown"):

        confounder, Z_sim, X_sim, Y_sim, X_star, Y_star = sim_IV_negbinomial(
            param.key, n=param.n, p=param.p,
            num_inst=param.num_inst, mu_c=param.mu_c, c_X=param.c_X,
            alpha0=param.alpha0, alphaT=param.alphaT, c_Y=param.c_Y,
            beta0=param.beta0, betaT=param.betaT, num_star=500, ps=param.ps)

        return confounder, Z_sim, X_sim, Y_sim, X_star, Y_star
    else:
        print("This combination is not implemented....")
