import logging
logging.basicConfig(level=logging.INFO)

import statsmodels.api as sm
import os

from skbio.stats.composition import ilr, ilr_inv
from sklearn.linear_model import LinearRegression
import jax
from statsmodels.sandbox.regression.gmm import IV2SLS
import numpy as onp
from jax.ops import index, index_update
from jax.lax import lgamma
from classo import classo_problem
from jax import numpy as np
from jax.scipy.special import digamma
from jax_minimize_wrapper import minimize
import skbio.stats.composition as cmp


import sys
sys.path.append(os.getcwd())


function_value = []


class LinearInstrumentModel:
    def __init__(self,
                 X_train,
                 Y_train,
                 Z_train,
                 params_init,
                 lambda_dirichlet,
                 max_iter=20):

        self.params_init_1 = params_init  # should ne a dictionary of alpha0 and alphaT
        self.params_alpha = None  # parameter for alpha estimation
        self.lambda_alpha = None
        self.X = X_train  # should be compositional in our context
        self.Y = Y_train  # outcome
        self.Z = Z_train  # instrument

        self.lambda_dirichlet = lambda_dirichlet  # should be a vector to be able to test more than one
        self._lambda_dirichlet = None  # int which is set in the optimization loop
        self.max_iter = max_iter

        self.n, self.p = X_train.shape  # with n being dimensi
        #self.X_n0 = X_train if X_train.sum()==X_train.shape[0] else self.add_term_to_zeros(self.X)  # add zeros to the term to be able to perform any log transformation
        self.X_n0 = self.add_term_to_zeros(self.X)  # add zeros to the term to be able to perform any log transformation
        self.Xhat = None

        self.params = None

    # setter function
    def set_lambda_dirichlet(self, lam):
        """set lambda in the optimization loop"""
        self._lambda_dirichlet = lam

    # method section
    @staticmethod
    def add_term_to_zeros(mat, axis=0, add_term=0.0001):
        """add a small term to the zeros along a specific axis"""
        # TODO : how to implement that sensibly in JAX
        X_n0 = np.zeros(mat.shape)
        for rr in range(len(mat)):
            if mat[rr, :].min() == 0:
                X_n0 = index_update(X_n0, index[rr, :],  (mat[rr, :] + add_term) / np.sum(mat[rr, :] + add_term))
            else:
                X_n0 = index_update(X_n0, index[rr, :], mat[rr, :])
        return X_n0


    def gradient_fun_alpha(self, alpha):
        """ this is the gradient of the objective function for beta """
        alpha0 = alpha[:self.p]
        alphaT = alpha[self.p:]
        g = np.exp((alpha0 * np.ones((self.n,1))) + (alphaT * self.Z[..., np.newaxis]))
        digamma_vec = np.vectorize(digamma)

        # derivation for alpha0
        com1 = digamma(np.sum(g, axis=1)) * np.sum(g, axis=1)
        com2 = np.sum(np.sum(np.log(self.X_n0) * (g - 1) - digamma_vec(g) * g, axis=1), axis=0)

        vec = np.tile(digamma_vec(np.sum(g, axis=1)), (self.p, 1)).T - digamma_vec(g) + np.log(self.X_n0)
        # TODO : dimensions of S are not working out
        S = np.dot((vec * g).T, self.Z)

        der = onp.zeros((len(alphaT), 1))
        alpha_pen = alphaT

        #if np.sum(alpha_pen != 0) > 0:
        #    der[np.where(alpha_pen != 0)] = self._lambda_dirichlet * alpha_pen[alpha_pen != 0] / np.abs(alpha_pen[alpha_pen != 0])
        der = self._lambda_dirichlet * np.sign(alpha_pen) * np.abs(alpha_pen)
        # TODO : in the actual function file this was S[,1] -> WHY?
        grad = np.vstack([-S, -S + der.T])

        return grad.flatten("F")


    def likelihood2(self, params):
        """ log likelihood function for dirichlet regression """
        alpha0 = params["alpha0"]   # p x 1
           # p x 1 -> would just need to add more parameters for each new instrument...?

        if self.Z.ndim < 2:
            alphaT = params["alphaT"]
            g = np.exp((alpha0 * np.ones((self.n,))).T + (alphaT * self.Z).T)  # n x p
        else:
            helperZ = [params["alphaT" + str(inst)] * self.Z[:, inst] for inst in range(self.Z.shape[1])]
            g = np.exp((alpha0 * np.ones((self.n,))).T + (sum(helperZ)).T)
        lgamma_vec = np.vectorize(lgamma)
        # compute log likelihood
        log_likelihood = np.sum(lgamma_vec(np.sum(g, axis=1)) +
                                np.sum((g - 1) * np.log(self.X_n0) - lgamma_vec(g), axis=1))
        return log_likelihood


    def likelihood2_lassopenalty(self, params):
        """ log likelihood function for the dirichlet regression with lasso penalty for variable selection """
        if self.Z.ndim < 2:
            alphaT = params["alphaT"]
        else:
            alphaT = np.hstack([params["alphaT" + str(inst)] for inst in range(self.Z.shape[1])])
        log_likelihood_penalty = -self.likelihood2(params) + self._lambda_dirichlet * (np.sum(np.abs(alphaT)))
        return log_likelihood_penalty


    def fit_dirichlet(self, lower_bound=None, upper_bound=None, verbose=False):
        """ parameter estimation by a dirichlet regression, estimate alpha coefficient """
        # TODO : check out which bounds they are actually setting. Seems like they take an estimate from the Dirichlet
        #  MLE for some of the beta
        #if lower_bound is None:
        #    lower_bound = np.hstack([self.beta_init[:self.p], -50 * np.ones((1 + self.q) * self.p)])
        #if upper_bound is None:
        #    upper_bound = np.hstack([self.beta_init[:self.p], 50 * np.ones((1 + self.q) * self.p)])

        # TODO : for each dimension in the middle variable I need an extra regression. (at least if I want to allow for
        #  more than one IV)
        BIC = {}
        estimates = {}
        num_inst = 0 if self.Z.ndim < 2 else self.Z.shape[1]

        # set the parameters for the iteration
        params = {"alpha0": self.params_init_1["alpha0"]}
        params.update({"alphaT" + str(inst): self.params_init_1["alphaT" + str(inst)] for inst in range(num_inst)})

        for lam in self.lambda_dirichlet:
            self.set_lambda_dirichlet(lam)
            res_alpha = minimize(self.likelihood2_lassopenalty,
                                 params,
                                 method="SLSQP",
                                 #bounds=bounds,
                                 options={'ftol': 1e-4, 'disp': verbose, "maxiter": self.max_iter})
            # TODO : insert selection of best regularization parameter
            # collect all successful models
            if res_alpha.success:
                params_est = {key: np.round(res_alpha.x[key], 3) for key in res_alpha.x.keys()}
                if self.Z.ndim < 2:
                    alphaT = params_est["alphaT"]
                else:
                    alphaT = np.hstack([params_est["alphaT" + str(inst)] for inst in range(self.Z.shape[1])])
                BIC_alpha = np.log(self.n) * (np.sum(params_est["alpha0"] != 0) + np.sum(alphaT != 0)) - \
                            2 * self.likelihood2(params_est)
                BIC.update({lam: BIC_alpha})
                estimates.update({lam: params_est})
        # choose the best model with the best regularization parameter
        if len(BIC) != 0:
            lam_best = min(BIC, key=BIC.get)
            self.params_alpha = estimates[lam_best]
            self.alpha_lambda = lam_best
            return BIC, estimates, self.params_alpha

    def predict_dirichlet(self, Z=None):
        """make prediction from estimated parameters"""
        if Z is None:
            Z = self.Z

        alpha0 = self.params_alpha["alpha0"]  # p x 1
        # p x 1 -> would just need to add more parameters for each new instrument...?

        if Z.ndim < 2:
            alphaT = self.params_alpha["alphaT"]
            g = np.exp((alpha0 * np.ones((self.n,))).T + (alphaT * Z).T)  # n x p
        else:
            helperZ = [self.params_alpha["alphaT" + str(inst)] * Z[:, inst] for inst in range(Z.shape[1])]
            g = np.exp((alpha0 * np.ones((Z.shape[0],))).T + (sum(helperZ)).T)

        g_sum = np.sum(g, axis=1)
        Xhat = np.apply_along_axis(lambda x: x/g_sum, 0, g)
        self.Xhat = Xhat
        return Xhat

    def fit_log_contrast_fast(self, X=None, threshold=0.7):
        """fit log contrast regression with Chris optimization package"""

        if X is None:
            X=self.Xhat
            # TODO : check if this is feasible
            X=self.add_term_to_zeros(X)
        # set up problem formulation
        # TODO : give labels -> output is give opt.solution.StabSel.label
        # TODO : for R1 -> this is tested, the rest try out (but should work)
        # TODO : for more parameters PATH might work
        # not working
        X_incl_inter = onp.hstack([onp.array(np.log(X))])
        C = onp.ones((1, len(X_incl_inter[0])))

        opt = classo_problem(X_incl_inter, onp.array(np.squeeze(self.Y)), onp.array(C))
        # specification of minimzation problem
        opt.formulation.intercept = True
        opt.formulation.huber = False
        opt.formulation.concomitant = False
        opt.formulation.classification = False
        opt.model_selection.PATH = True  #
        # opt.model_selection.LAMfixed independent of StabSel
        opt.model_selection.StabSel = True  # Choice, (before it was set to true)
        opt.model_selection.StabSelparameters.method = 'lam'
        # opt.model_selection.StabSelparameters.threshold_label = 0.5

        opt.model_selection.StabSelparameters.threshold = threshold

        opt.solve()
        beta = opt.solution.StabSel.refit
        try:
            print(opt.solution)
        except:
            print("Error in computation")
        #beta = opt.solution.PATH.BETAS  # this has to be done when StabSel is False -> no refitting necessary, but Path
        # save parameters for prediction

        self.params_beta = {"beta0": np.round(beta[0], 3),
                            "betaT": np.round(beta[1:], 3)}

        return opt

    def predict_log_contrast(self, X=None):
        """predict from the estimated log contrast model"""
        if X is None:
            X = self.Xhat
        beta0 = self.params_beta["beta0"]
        betaT = self.params_beta["betaT"]
        # TODO : make X with no zeros
        Yhat = (beta0 * np.ones((X.shape[0],))).T + np.dot(np.log(X), betaT)
        return Yhat




class ALR_Model:
    def __init__(self):
        self.alpha = None
        self.m_0 = None


    def fit(self, X_sim, Z_sim):
        """fit model from mediation paper """
        p = X_sim.shape[1]
        ZZ_sim = sm.add_constant(Z_sim)
        #M = np.log(X_sim)[:, :p - 1]
        M = cmp.alr(X_sim)
        res = np.linalg.lstsq(ZZ_sim, M)[0]
        #alt_m_0, alt_alpha = np.linalg.lstsq(ZZ_sim, M)[0]
        if res.ndim > 1:
            alt_m_0 = res[0, :]
            alt_alpha = res[1:, :]
        else:
            alt_m_0, alt_alpha = res
        # transformation of alt parameters
        self.alpha = cmp.alr_inv(alt_alpha)

        # fit some baseline distribution
        self.m_0 = cmp.alr_inv(alt_m_0)

    def predict(self, Z_sim):
        """predict from estimated parameters """
        n = Z_sim.shape[0]
        p = len(self.m_0)

        if Z_sim.squeeze().ndim > 1:
            Xhat = cmp.alr_inv(cmp.alr(self.m_0) * np.ones((n, p - 1)) + Z_sim@cmp.alr(self.alpha))
        else:
            Xhat = cmp.alr_inv(cmp.alr(self.m_0) * np.ones((n, p - 1)) + cmp.alr(self.alpha) * Z_sim)
        return Xhat


def dirichlet_logcontrast(Z, X, Y, Xstar, mle, lambda_dirichlet, max_iter, logcontrast_threshold):
    """Model with 1st stage Dirichlet Regression and 2nd stage LogContrast Regression"""
    key = jax.random.PRNGKey(1991)
    n, p = X.shape
    n, num_inst = Z.shape


    params_init = {
       "alpha0": mle[..., np.newaxis],  # np.abs(jax.random.normal(key, (p,1))),
    }
    params_init.update({"alphaT" + str(inst): np.abs(jax.random.normal(key, (p, 1))) for inst in range(num_inst)})
    LinIVModel_sim = LinearInstrumentModel(X, Y, Z, params_init,
                                                  lambda_dirichlet=lambda_dirichlet,
                                                  max_iter=max_iter)

    # 1st Stage
    LinIVModel_sim.fit_dirichlet()
    #print(LinIVModel_sim.params_alpha)
    Xhat = LinIVModel_sim.predict_dirichlet()

    # 2nd Stage
    opt = LinIVModel_sim.fit_log_contrast_fast(threshold=logcontrast_threshold)
    Yhat = LinIVModel_sim.predict_log_contrast(Xstar)

    beta_est_0, beta_est_T = LinIVModel_sim.params_beta.values()
    beta = np.hstack([beta_est_0, beta_est_T])

    print("Beta DirichLetLogContrast: " + str(beta_est_0) + str(beta_est_T))

    return beta, Xhat, Yhat


def ilr_noregression(Z, X, verbose=False):
    """Model with 2SLS approach but with a ilr transformed variable"""
    # use 2SLS on ilr transformation
    n, p = X.shape
    Z_2sls = sm.add_constant(onp.array(Z))
    X_n0 = LinearInstrumentModel.add_term_to_zeros(X)
    X_ilr = ilr(onp.array(X_n0))
    first = sm.OLS(X_ilr, Z_2sls).fit()
    Xhat = first.predict(Z_2sls)
    alpha = first.params
#    #V = cmp._gram_schmidt_basis(p)
    return alpha, cmp.ilr_inv(Xhat)

def noregression_ilr(X, Y, Xstar, verbose):
    """Model with 2SLS approach but with a ilr transformed variable"""
    # use 2SLS on ilr transformation
    n, p = X.shape
    X_n0 = LinearInstrumentModel.add_term_to_zeros(X)
    X_ilr = sm.add_constant(ilr(onp.array(X_n0)))
    X_star_ilr = sm.add_constant(ilr(onp.array(Xstar)))
    second = sm.OLS(onp.array(Y), X_ilr).fit()
    if verbose:
        print(second.summary())
    Yhat = second.predict(X_star_ilr)
    beta = second.params
    #V = cmp._gram_schmidt_basis(p)
    return beta, Yhat


def ilr_ilr(Z, X, Y, Xstar):
    """Model with 2SLS approach but with a ilr transformed variable"""
    # use 2SLS on ilr transformation
    n, p = X.shape
    Z_2sls = sm.add_constant(onp.array(Z))
    X_n0 = LinearInstrumentModel.add_term_to_zeros(X)
    X_ilr = sm.add_constant(ilr(onp.array(X_n0)))
    Y_2sls = onp.array(Y)

    iv2sls_log = IV2SLS(Y_2sls, X_ilr, instrument=Z_2sls).fit()
    print(iv2sls_log.summary())
    Yhat = iv2sls_log.predict(sm.add_constant(ilr(onp.array(Xstar))))
    V = cmp._gram_schmidt_basis(p)
    beta = np.hstack([iv2sls_log.params[0], V.T @ iv2sls_log.params[1:]])

    return beta, Yhat


def ilr_logcontrast(Z, X, Y, Xstar, logcontrast_threshold):
    """Model with 1st stage as OLS regression on ilr transformed components and 2nd stage as log contrast model"""

    Z_2sls = sm.add_constant(onp.array(Z))
    X_n0 = LinearInstrumentModel.add_term_to_zeros(X)
    X_ilr = ilr(onp.array(X_n0))

    # 1st Stage
    reg = LinearRegression().fit(Z_2sls, X_ilr)
    X_fitted_ilr = reg.predict(Z_2sls)
    Xhat = ilr_inv(X_fitted_ilr)

    LinIVModel_ilr = LinearInstrumentModel(Xhat, Y, None, None,
                                                   lambda_dirichlet=None,
                                                   max_iter=None)
    opt_ilr = LinIVModel_ilr.fit_log_contrast_fast(Xhat, threshold=logcontrast_threshold)
    Yhat = LinIVModel_ilr.predict_log_contrast(Xstar)

    beta_ilr_0, beta_ilr_T = LinIVModel_ilr.params_beta.values()
    # back transformation of ilr beta regression parameters

    beta = np.hstack([beta_ilr_0, beta_ilr_T])

    return beta, Xhat, Yhat


def noregression_logcontrast(Z, X, Y, Xstar, logcontrast_threshold):
    """
    Benchmark model which ignores the instrumental variable approach and fits a log contrast regression with classo
    from X to Y
    """
    X_n0 = LinearInstrumentModel.add_term_to_zeros(X)
    LinIVModel_benchmark = LinearInstrumentModel(X, Y, None, None,
                                                         lambda_dirichlet=None,
                                                         max_iter=None)
    opt_benchmark = LinIVModel_benchmark.fit_log_contrast_fast(X_n0, threshold=logcontrast_threshold)
    Yhat = LinIVModel_benchmark.predict_log_contrast(Xstar)
    beta_bench_0, beta_bench_T = LinIVModel_benchmark.params_beta.values()
    beta = np.hstack([beta_bench_0, beta_bench_T])

    return beta, Yhat
