
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
#sys.path.append(os.path.join(os.getcwd(), "generalIVmodel/"))


import boundiv
import kiv
from simulate_data_fct import sim_IV_ilr_linear, sim_IV_ilr_nonlinear, sim_IV_lognormal_linear, sim_IV_ilr_linear_normalZ
from method_fct import noregression_ilr, noregression_logcontrast, dirichlet_logcontrast, ilr_logcontrast, ALR_Model
from method_fct import ilr_ilr, ilr_noregression

from helper_fct import *

from plot_fct import plot_mse_results, plot_beta_results
from method_fct import LinearInstrumentModel




def run_methods_confidence_interval(key, num_iteration,
                                    n, p, num_inst, mu_c, c_X, alpha0, alphaT, c_Y, beta0, betaT,
                                    is_nonlinear=False, is_lognormal=False, is_ZNormal=False,
                                    fit_kiv=True,
                                    num_star=200, logcontrast_threshold=0.7, max_iter=500,
                                    lambda_dirichlet=np.array([0.1, 1, 2, 5, 10])):
    """ run models for different seeds """

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
        elif is_lognormal:
            confounder, Z_sim, X_sim, Y_sim, X_star, Y_star = sim_IV_lognormal_linear(key,
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
        elif is_ZNormal:
            confounder, Z_sim, X_sim, Y_sim, X_star, Y_star = sim_IV_ilr_linear_normalZ(
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

        if p < 10:
            try:
                mle = dirichlet.mle(X_sim[(np.abs(Z_sim)<=0.2).sum(axis=1)==p,:],
                                    tol=.001)
            except:
                mle = np.ones((p,)) / p
        else:
            mle = np.ones((p,)) / p



        mse_all, beta_all, title_all, mse_large = run_methods_all(Z_sim, X_sim, Y_sim, X_star, Y_star, betaT,
                        mle, lambda_dirichlet, max_iter, logcontrast_threshold,
                        fit_kiv=fit_kiv)

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

    #betaT_log = V.T @betaT

    fig_mse = plot_mse_results(df_mse)
    fig_mse.show()
    fig_beta = plot_beta_results(df_beta, betaT)
    fig_beta.show()


    return df_beta, df_mse, mse_large_confidence


def run_methods_all(Z_sim, X_sim, Y_sim, X_star, Y_star, beta_ilr_true,
                    mle, lambda_dirichlet, max_iter, logcontrast_threshold,
                   fit_kiv=False):
    """Run all available methods to compare the performance"""
    n, p = X_sim.shape
    V = cmp._gram_schmidt_basis(p)
    mse_all = []
    title_all = []
    beta_all = []

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

    if fit_kiv:
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
        title = "KIVwhole"
        title_all.append(title)
        print("")

        print("---------------------------------------------------------------------------------------------")
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<< ONLY SECOND STAGE - Kernel Regression KIV >>>>>>>>>>>>>>>>>>>>>>>>>")

        kernel="linear"
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
        title = "KIV+KIV"
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



    if any(i>40 for i in mse_all[3:]):
        mse_large = {"X_sim": X_sim,
                     "Y_sim": Y_sim,
                     "X_star": X_star,
                     "Y_star": Y_star}
    else:
        mse_large = None

    return mse_all, beta_all, title_all, mse_large


def run_diversity_estimation_methods(Z, X, Y, Ytrue=None, methods=["OLS", "2SLS", "KIV", "Bounds"]):
    """take diversity estimates and run all available methods"""

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











