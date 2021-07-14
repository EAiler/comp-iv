
import logging
logging.basicConfig(level=logging.INFO)


import os
import jax
import numpy as onp
import pandas as pd
import dirichlet
import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import IV2SLS
import skbio.stats.composition as cmp



import sys
sys.path.append(os.getcwd())


import boundiv
import kiv
from simulate_data_fct import sim_IV_ilr_linear, sim_IV_ilr_nonlinear, sim_IV_negbinomial
from method_fct import noregression_ilr, noregression_logcontrast, dirichlet_logcontrast, ilr_logcontrast, ALR_Model
from method_fct import ilr_ilr, ilr_noregression

from helper_fct import *

from plot_fct import plot_mse_results, plot_beta_results
from method_fct import LinearInstrumentModel




def run_methods_confidence_interval(key, num_iteration,
                                    n, p, num_inst, mu_c, c_X, alpha0, alphaT, c_Y, beta0, betaT,
                                    is_nonlinear=False, is_negbinom=False,
                                    num_star=200, logcontrast_threshold=0.7, max_iter=500,
                                    lambda_dirichlet=np.array([0.1, 1, 2, 5, 10])):
    """Run all available methods to compare the performance

    Parameters
    ----------
    key : jax PRNG key
    num_iteration : int
        number of iterations for computation of the confidence interval
    n : int
        number of samples for the IV model
    p : int
        number of microbiota in X
    num_inst : int
        number of instruments
    mu_c : np.ndarray / int
        different interpretation:
        for negative binomial, this is the compositional confounding vector;
        for linear, this is the mean of the confounder
    c_X  : int
        different interpretation:
        for negative binomial, this is the dispersion parameter for the negative binomial distribution;
        for linear, this is the standard deviation of the confounder
    alpha0 : float
        intercept of the relationship between Z and X
    alphaT : np.ndarray
        slope of the relationship between Z and X
    c_Y : np.ndarray / int
        different interpretation:
        for negative binomial, vector of how composition influence Y;
        for linear, int of how confounder influences Y
    beta0 : float
        intercept of causal relationship of X on Y
    betaT : np.ndarray
        slope of causal relationship of X on Y
    is_nonlinear : bool
        if relationship of X and Y should be modelled as a non-linear function
    is_negbinom : bool
        if relationship of Z on X should be modelled within a zero-inflated negative binomial model
    num_star : int=200
        dimension of interventional X that should be created
    logcontrast_threshold : float=0.7
        hyperparameter for the log contrast regression, between 0 and 1
    max_iter : int=500
        hyperparameter for log contrast regression, maximum number of iterations to find a solution
    lambda_dirichlet : np.ndarray=np.array([0.1, 1, 2, 5, 10])
        hyperparameter for the dirichlet regression, provides array of penalty lambda parameters to choose from


    Returns
    -------
    df_beta : pd.DataFrame
        dataframe containing all beta results
    df_mse: pd.DataFrame
        dataframe containing all oos mse results
    mse_large_confidence : dict
        dictionary containing the iterations of the confidence interval where the oos mse was unnaturally
        large for debugging
    """

    mse_confidence = []
    title_confidence = []
    beta_confidence = []
    mse_large_confidence = {}
    for iter in range(num_iteration):
        print("**************************************************************************************************")
        print("*****************************************"+"We are at "+str(iter)+" of "+str(num_iteration)+"***********************************************")
        print("**************************************************************************************************")

        # generate new seeds
        key, subkey = jax.random.split(key)
        if is_nonlinear:
            confounder, Z_sim, X_sim, Y_sim, X_star, Y_star = sim_IV_ilr_nonlinear(
                key,
                n=n,
                p=p,
                num_inst=num_inst,
                mu_c=mu_c,
                c_X=c_X,
                alpha0=alpha0,
                alphaT=alphaT,
                c_Y=c_Y,
                beta0=beta0,
                betaT=betaT,
                num_star=num_star)
        elif is_negbinom:
            confounder, Z_sim, X_sim, Y_sim, X_star, Y_star = sim_IV_negbinomial(key,
                                    n=n,
                                    p=p,
                                    num_inst=num_inst,
                                    mu_c=mu_c,
                                    c_X=c_X,
                                    alpha0=alpha0,
                                    alphaT=alphaT,
                                    c_Y=c_Y,
                                    beta0=beta0,
                                    betaT=betaT,
                                    num_star=num_star)
            logcontrast_threshold=0.3

        else:
            confounder, Z_sim, X_sim, Y_sim, X_star, Y_star = sim_IV_ilr_linear(
                subkey,
                n=n,
                p=p,
                num_inst=num_inst,
                mu_c=mu_c,
                c_X=c_X,
                alpha0=alpha0,
                alphaT=alphaT,
                c_Y=c_Y,
                beta0=beta0,
                betaT=betaT,
                num_star=num_star)

        # TODO : check if this is necessary or might mess up some stuff
        X_sim = LinearInstrumentModel.add_term_to_zeros(X_sim)
        X_star = LinearInstrumentModel.add_term_to_zeros(X_star)


        mse_all, beta_all, title_all, mse_large = run_methods_all(Z_sim, X_sim, Y_sim, X_star, Y_star, betaT,
                        lambda_dirichlet, max_iter, logcontrast_threshold)

        beta_confidence.append(beta_all)
        mse_confidence.append(mse_all)
        title_confidence.append(title_all)
        mse_large_confidence.update({iter: mse_large})

    V = cmp._gram_schmidt_basis(p)
    flatten = lambda t: [item for sublist in t for item in sublist]
    title_all = flatten(title_confidence)
    mse_all = flatten(mse_confidence)
    beta_all = flatten(beta_confidence)

    mse_dict = dict([(key, np.array([])) for key in set(title_all)])

    for i in range(len(title_all)):
        if mse_all[i] is not None:
            value = mse_dict[title_all[i]]
            mse_dict.update({title_all[i]: np.append(value, mse_all[i])})
        else:
            value = mse_dict[title_all[i]]
            mse_dict.update({title_all[i]: np.append(value, np.nan)})

    df_mse = pd.DataFrame(mse_dict)
    df_mse.dropna(axis=1, inplace=True)
    cat_order = df_mse.mean().sort_values().index
    df_mse = df_mse.reindex(cat_order, axis=1)
    df_mse = pd.melt(df_mse, var_name="Method", value_name="MSE")

    beta_all_2 = [V.T @ i if i is not None else np.repeat(np.nan, p) for i in beta_all]
    df_beta = pd.DataFrame(zip(title_all, beta_all_2), columns=["Method", "Beta"])

    fig_mse = plot_mse_results(df_mse)
    fig_mse.show()
    fig_beta = plot_beta_results(df_beta, betaT)
    fig_beta.show()


    return df_beta, df_mse, mse_large_confidence


def run_methods_all(Z_sim, X_sim, Y_sim, X_star, Y_star, beta_ilr_true,
                    lambda_dirichlet, max_iter, logcontrast_threshold, mse_large_threshold=40):
    """Run all available methods to compare the performance

    Parameters
    ----------
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
    mle : np.ndarray
        starting point for dirichlet regression in the first stage
    lambda_dirichlet : float
        penalizing lambda used for dirichlet regression in the first stage
    max_iter : int
        maximum numver if iterations for log contrast regression in the second stage
    mse_large_threshold : int=40
        all data that produces a oos mse over the threshold value is saved in a dictionary attached to mse_large for de-
        bugging purposes


    Returns
    -------
    mse_all : np.ndarray
        out of sample mean squared error for all methods that have been tested
    beta_all : np.ndarray
        beta values for all methods that have been tested
    title_all : np.ndarray
        array of strings which show which item in mse_all, beta_all belongs to which method
    mse_large : None or dict
        dictionary holding the data which produces a oos mse over the mse_large_threshold value
    """
    n, p = X_sim.shape
    V = cmp._gram_schmidt_basis(p)
    mse_all = []
    title_all = []
    beta_all = []

    # estimate starting point for dirichlet regression
    if p < 10:
        try:
            mle = dirichlet.mle(X_sim[(np.abs(Z_sim) <= 0.2).sum(axis=1) == p, :],
                                tol=.001)
        except:
            mle = np.ones((p,)) / p
    else:
        mle = np.ones((p,)) / p

    print("---------------------------------------------------------------------------------------------")
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< ONLY Second Stage - ILR Regression >>>>>>>>>>>>>>>>>>>>>>>>>>>")
    beta, Yhat = noregression_ilr(X_sim, Y_sim, X_star, verbose=False)
    mse = np.mean((Yhat - Y_star) ** 2)
    mse_all.append(mse)
    print("True Beta: " + str(beta_ilr_true))
    print("Estimated Beta: " + str(np.round(beta[1:], 2)))
    print("Estimated Beta: " + str(np.round(V.T@beta[1:], 2)))
    print("Error: " + str(np.round(mse, 2)))
    title = "ONLY Second ILR"
    title_all.append(title)
    beta_all.append(beta[1:])
    print("")

    print("---------------------------------------------------------------------------------------------")
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< ONLY Second Stage - Log Contrast >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    # NO REGRESSION LOG CONTRAST REGRESSION
    try:
        betahat, Yhat = noregression_logcontrast(Z_sim, X_sim, Y_sim, X_star,
                                                 logcontrast_threshold)
        mse = np.mean((Yhat - Y_star) ** 2)
        mse_all.append(mse)
        beta_all.append(V @ betahat[1:])
        print("True Beta: " + str(beta_ilr_true))
        print("Estimated Beta: " + str(np.round(betahat[1:], 2)))
        print("Error: " + str(np.round(mse, 2)))
        title = "ONLY Second LC"
        title_all.append(title)
        print("")
    except:
        print("No solution for " + str(title))


    print("---------------------------------------------------------------------------------------------")
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 2SLS - Dirichlet + Log Contrast >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    # DIRICHLET LOG CONTRAST REGRESSION
    if X_sim.shape[1] < 10:
        title = "2SLS - Dirichlet + Log Contrast"
        try:
            betahat, X_diri_log, Yhat = dirichlet_logcontrast(Z_sim, X_sim, Y_sim, X_star, mle, lambda_dirichlet, max_iter,
                                                              logcontrast_threshold)

            mse = np.mean((Yhat - Y_star) ** 2)
            mse_all.append(mse)
            beta_all.append(V @ betahat[1:])
            print("True Beta: " + str(beta_ilr_true))
            print("Estimated Beta: " + str(np.round(V @ betahat[1:], 2)))
            print("Error: " + str(np.round(mse, 2)))
            title = "DIR+LC"
            title_all.append(title)
            print("")
        except:
            print("No solution found for Dirichlet Regression")
            mse_all.append(None)
            title_all.append(title)
    else:
        print("Dirichlet Regression not tried for performance reasons.")


    print("---------------------------------------------------------------------------------------------")
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 2SLS - ILR + Log Contrast >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    # ILR REGRESSION LOG CONTRAST REGRESSION
    betahat, X_ilr_log, Yhat = ilr_logcontrast(Z_sim, X_sim, Y_sim, X_star,
                                               logcontrast_threshold)
    mse = np.mean((Yhat - Y_star) ** 2)
    mse_all.append(mse)
    beta_all.append(V @ betahat[1:])
    print("True Beta: " + str(beta_ilr_true))
    print("Estimated Beta: " + str(np.round(betahat[1:], 2)))
    print("Error: " + str(np.round(mse, 2)))
    title = "ILR+LC"
    title_all.append(title)
    print("")


    print("---------------------------------------------------------------------------------------------")
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< ALR MODEL>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    first_stage = ALR_Model()
    first_stage.fit(X_sim, Z_sim)
    Xhat_alr2 = first_stage.predict(Z_sim)

    betahat, Yhat = noregression_logcontrast(Z_sim, Xhat_alr2, Y_sim, X_star,
                                                         logcontrast_threshold)
    mse = np.mean((Yhat - Y_star) ** 2)
    mse_all.append(mse)
    beta_all.append(V @ betahat[1:])
    print("True Beta: " + str(beta_ilr_true))
    print("Estimated Beta: " + str(np.round(V @ betahat[1:], 2)))
    print("Error: " + str(np.round(mse, 2)))
    title = "ALR+LC"
    title_all.append(title)
    print("")

    print("---------------------------------------------------------------------------------------------")
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 2SLS - Kernel Regression KIV >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

    X_sim_ilr = cmp.ilr(X_sim)
    X_star_ilr = cmp.ilr(X_star)

    def whiten_data(X):
        mu, std = X.mean(axis=0), X.std(axis=0)
        X_std = (X - mu) / std
        return X_std, mu, std

    XX, mu_x, std_x = whiten_data(X_sim_ilr)
    YY, mu_y, std_y = whiten_data(Y_sim)
    ZZ, mu_z, std_z = whiten_data(Z_sim)

    _, Yhat = kiv.fit_kiv(ZZ, XX, YY, xstar=((X_star_ilr - mu_x) / std_x))
    Yhat = std_y * Yhat + mu_y
    mse = np.mean((Yhat - Y_star) ** 2)
    mse_all.append(mse)
    beta_all.append(None)
    print("Error: " + str(np.round(mse, 2)))
    title = "KIV"
    title_all.append(title)
    print("")

    print("---------------------------------------------------------------------------------------------")
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<< ONLY SECOND STAGE - Kernel Regression KIV >>>>>>>>>>>>>>>>>>>>>>>>>")

    kernel = "linear"
    from sklearn.kernel_ridge import KernelRidge
    reg = KernelRidge(kernel=kernel).fit(XX, YY)
    Yhat = reg.predict((X_star_ilr - mu_x) / std_x)
    Yhat = std_y * Yhat + mu_y
    mse = np.mean((Yhat - Y_star) ** 2)
    mse_all.append(mse)
    beta_all.append(None)
    print("Error: " + str(np.round(mse, 2)))
    title = "ONLY Second KIV"
    title_all.append(title)
    print("")


    print("---------------------------------------------------------------------------------------------")
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<< 2SLS - Kernel Regression KIV (manual) >>>>>>>>>>>>>>>>>>>>>>>>>")

    reg = KernelRidge(kernel=kernel).fit(ZZ, XX)
    Xhat_ilr = reg.predict(ZZ)
    reg2 = KernelRidge(kernel=kernel).fit(Xhat_ilr, YY)
    Yhat = reg2.predict((X_star_ilr - mu_x) / std_x)
    Yhat = std_y * Yhat + mu_y
    mse = np.mean((Yhat - Y_star) ** 2)
    mse_all.append(mse)
    beta_all.append(None)
    print("Error: " + str(np.round(mse, 2)))
    title = "Kernel+Kernel"
    title_all.append(title)
    print("")

    print("---------------------------------------------------------------------------------------------")
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 2SLS - ILR ILR Regression Implementation >>>>>>>>>>>>>>>>>>>>>>>")
    # ILR REGRESSION ILR REGRESSION
    betahat, Yhat = ilr_ilr(Z_sim, X_sim, Y_sim, X_star)
    mse = np.mean((Yhat - Y_star) ** 2)
    mse_all.append(mse)
    beta_all.append(V @ betahat[1:])
    print("True Beta: " + str(beta_ilr_true))
    print("Estimated Beta: " + str(np.round(V @ betahat[1:], 2)))
    print("Error: " + str(np.round(np.mean((Yhat - Y_star) ** 2), 2)))
    title = "ILR+ILR"
    title_all.append(title)
    print("")



    if any(i>mse_large_threshold for i in mse_all[3:]):
        mse_large = {"X_sim": X_sim,
                     "Y_sim": Y_sim,
                     "X_star": X_star,
                     "Y_star": Y_star}
    else:
        mse_large = None

    return mse_all, beta_all, title_all, mse_large


def run_diversity_estimation_methods(Z, X, Y, Ytrue=None, methods=["OLS", "2SLS", "KIV", "Bounds"]):
    """take diversity estimates and run all available methods, prints out first stage F-test to give an indication
    of instrument strength

    Parameters
    ----------
    Z : np.ndarray
        matrix of instrument entries
    X : np.ndarray
        matrix of diversity estimates
    Y : np.ndarray
        matrix of weight entries
    Ytrue : np.ndarray
        matrix of causal weight entries, are standardized the same way as Y is
    methods : list=["OLS", "2SLS", "KIV", "Bounds"]
        list of methods that should be run on the data


    Results
    -------
    x : np.ndarray
        standardized matrix of diversity estimates
    y : np.ndarray
        standardized matrix of outcome entries
    Ytrue : np.ndarray
        standardized matrix of outcome entries for causal Ytrue input
    xstar : np.ndarray
        interventional x value for true causal effect
    xstar_bound : np.ndarray
        interventional x value for bound evaluation
    ystar_ols : np.ndarray
        outcome when estimating causal effect via OLS
    ystar_2sls : np.ndarray
        outcome when estimating causal effect via 2SLS
    ystar_kiv : np.ndarray
        outcome when estimating causal effect via KIV
    results : np.ndarray
        outcome of bounding method

    """

    def whiten_data(col):
        mu = col.mean()
        std = col.std()
        return (col - mu) / std, mu, std

    # whiten the data
    Z = Z  # whiten_data(Z)
    X, _, _ = whiten_data(X)
    Y, mu_y, mu_std = whiten_data(Y)
    if Ytrue is not None:
        Ytrue = (Ytrue - mu_y)/mu_std

    z = onp.array(Z).squeeze()
    y = onp.array(Y).squeeze()
    x = onp.array(X).squeeze()

    zz = sm.add_constant(z)
    xx = sm.add_constant(x)

    xstar = np.linspace(x.min(), x.max(), 30)

    # F Test
    ols1 = sm.OLS(x, zz).fit()
    xhat = ols1.predict(zz)
    print(ols1.summary())


    # OLS
    if "OLS" in methods:
        ols = sm.OLS(y, xx).fit()
        coeff_ols = ols.params
        ystar_ols = coeff_ols[0] + coeff_ols[1] * xstar
    else:
        ystar_ols = None

    # 2SLS
    if "2SLS" in methods:
        iv2sls = IV2SLS(y, xx, zz).fit()
        coeff_2sls = iv2sls.params
        ystar_2sls = coeff_2sls[0] + coeff_2sls[1] * xstar
    else:
        ystar_2sls = None

    # KIV
    if "KIV" in methods:
        xstar, ystar_kiv = kiv.fit_kiv(z, x, y, xstar=xstar)
    else:
        ystar_kiv = None

    # BoundIV
    if "Bounds" in methods:
        xstar_bound = np.linspace(np.quantile(x, 0.1), np.quantile(x, 0.9), 5)
        satis, results = boundiv.fit_bounds(z, x, y, xstar_bound)
    else:
        xstar_bound = None
        results = None

    return x, y, Ytrue, xstar, xstar_bound, ystar_ols, ystar_2sls, ystar_kiv, results











