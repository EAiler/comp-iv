# import statements
import os

from typing import Tuple, Text
from absl import logging
import jax.numpy as np
from jax import random, value_and_grad, jit
import jax.experimental.optimizers as optim
from jax.ops import index_update
from tqdm import tqdm

import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "generalIVmodel/"))
sys.path.append(os.path.join(os.getcwd(), "generalIVmodel/src"))

import utils_giv as utils_giv

ArrayTup = Tuple[np.ndarray, np.ndarray]
Params = Tuple[np.ndarray, np.ndarray, np.ndarray]

# =============================================================================
# DEFAULT PARAMETER SETTINGS
# =============================================================================
num_data = 5000
response_type = "poly" #("poly", "gp", "mlp")
num_xstar = 15
dim_theta = 2
 # number of grid points -> default was 20
num_rounds = 150
opt_steps = 30
bs = 1024
bs_constr = 4096
slack = 0.2
slack_abs = 0.2
lr = 0.001
decay_steps = 1000
decay_rate = 1.0
staircase = False
momentum = 0.9
tau_init = 0.1
tau_factor = 1.08
tau_max = 1.0

run_2sls = False
run_kiv = False
seed = 0

plot_init = False
plot_intermediate = False
plot_final = True
store_data = False

# =============================================================================
# RHS CONSTRAINT FUNCTIONS THAT MUST BE OVERWRITTEN
# =============================================================================
@jit
def get_phi(y: np.ndarray) -> np.ndarray:
  """The phis for the constraints."""
  return np.array([np.mean(y, axis=-1), np.var(y, axis=-1)]).T


@jit
def get_rhs(thetahat: np.ndarray, xhats_pre: np.ndarray) -> np.ndarray:
  """Construct the RHS for the second approach (unsing basis functions phi)."""
  return get_phi(response(thetahat, xhats_pre))


def make_constraint_lhs(y: np.ndarray,
                        bin_ids: np.ndarray,
                        z_grid: np.ndarray) -> np.ndarray:
  """Get the LHS of the constraints."""
  # Use indicator annealing approach
  tmp = []
  for i in range(len(z_grid)):
    bool_index = tuple([bin_ids == i])
    tmp.append(get_phi(y[bool_index]))
  tmp = np.array(tmp)

  # Smoothen LHS constraints with UnivariateSpline smoothing
  logging.info(f"Smoothen constraints with splines. Fixed factor: 0.2 ...")
  lhs = []
  for i in range(tmp.shape[-1]):

    lhs.append(tmp[:, i])
  #  spl = UnivariateSpline(z_grid, tmp[:, i], s=0.2)
    #lhs.append(spl(z_grid))

  lhs = np.array(lhs).T
  return lhs


@jit
def response_poly(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
  """The response function."""
  return np.polyval(theta, x)



# Must be overwritten with one of the available response functions
# noinspection PyUnusedLocal
@jit
def response(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
  """The response function."""
  return np.empty((0,))


# =============================================================================
# OPTIMIZATION (AUGMENTED LAGRANGIAN)
# =============================================================================

@jit
def get_constraint_term(constr: np.ndarray,
                        lmbda: np.ndarray,
                        tau: float) -> float:
  """Compute the sum of \psi(c_i, \lambda, \tau) for the Lagrangian."""
  case1 = - lmbda * constr + 0.5 * tau * constr**2
  case2 = - 0.5 * lmbda**2 / tau
  psi = np.where(tau * constr <= lmbda, case1, case2)
  return np.sum(psi)


@jit
def update_lambda(constr: np.ndarray,
                  lmbda: np.ndarray,
                  tau: float) -> np.ndarray:
    """Update Lagrangian parameters lambda."""
    return np.maximum(lmbda - tau * constr, 0)


@jit
def make_cholesky_factor(l_param: np.ndarray) -> np.ndarray:
  """Get the actual cholesky factor from our parameterization of L."""
  lmask = np.tri(l_param.shape[0])
  lmask = index_update(lmask, (0, 0), 0)
  tmp = l_param * lmask
  idx = np.diag_indices(l_param.shape[0])
  return index_update(tmp, idx, np.exp(tmp[idx]))


@jit
def make_correlation_matrix(l_param: np.ndarray) -> np.ndarray:
    """Get correlation matrix from our parameterization of L."""
    chol = make_cholesky_factor(l_param)
    return chol @ chol.T


@jit
def objective_rhs_psisum_constr(
    key: np.ndarray,
    params: Params,
    lmbda: np.ndarray,
    tau: float,
    lhs: np.ndarray,
    slack: np.ndarray,
    xstar: float,
    tmp_pre: np.ndarray,
    xhats_pre: np.ndarray,
) -> Tuple[float, np.ndarray, float, np.ndarray]:
  """Estimate the objective, RHS, psisum (constraint term), and constraints.

  Refer to the docstring of `lagrangian` for a description of the arguments.
  """

  response = response_poly
  # (k+1, k+1), (k,), (k,)
  L, mu, log_sigma = params
  n = tmp_pre.shape[-1]
  # (k, n)
  tmp = random.normal(key, (dim_theta, n))
  # (k+1, n)
  tmp = np.concatenate((tmp_pre, tmp), axis=0)
  # (k+1, n) add initial dependence

  #print(make_cholesky_factor(L) @ tmp)
  tmp = utils_giv.std_normal_cdf(make_cholesky_factor(L) @ tmp)
  # (k, n) get thetas with current means and variances
  thetahat = utils_giv.normal_cdf_inv(tmp[1:, :], mu, log_sigma)
  # (1,) main objective <- (n,) <- (k, n), ()
  obj = np.mean(response(thetahat, np.array(xstar)))
  # (m, l) computes rhs for all z
  #rhs = get_rhs(thetahat, xhats_pre)
  # TODO : changed to directly get phihat
  rhs = get_phi(response(thetahat, xhats_pre))

  # (m * l,) constraints (with tolerances)
  constr = slack - np.ravel(np.abs(lhs - rhs))
  # (1,) constraint term of lagrangian
  psisum = get_constraint_term(constr, lmbda, tau)
  return obj, rhs, psisum, constr


@jit
def lagrangian(key: np.ndarray,
               params: Params,
               lmbda: np.ndarray,
               tau: float,
               lhs: np.ndarray,
               slack: np.ndarray,
               xstar: float,
               tmp_pre: np.ndarray,
               xhats_pre: np.ndarray,
               sign: float = 1.) -> float:
  """Estimate the Lagrangian at a given \eta.

  For given $\eta$ compute MC estimate of the Lagrangian with samples from
  $p(\theta | x, z)$, which are used for the constraints, but also
  (marginalized) for the main objective.

  Args:
      key: Key for the random number generator.
      params: A 3-tuple with the parameters to optimize consisting of
          L: Lower triangular matrix from which we compute the Cholesky factor.
              (Not the Cholesky factor itself!).
              Dimension: (DIM_THETA + 1, DIM_THETA + 1)
          mu: The means of the (Gaussian) marginals of the thetas.
              Dimension: (DIM_THETA, )
          log_sigma: The log of the standard deviations of the (Gaussian)
              marginals of the thetas. (Use log to ensure they're positive).
              Dimension: (DIM_THETA, )
      lmbda: The Lagrangian multipliers lambda. Dimension: (NUM_Z * NUM_PHI, )
      tau: The temperature parameter for the augmented Lagrangian approach.
      lhs: The LHS of the constraints. Dimension: (NUM_Z, NUM_PHI)
      slack: The tolerance for how well the constraints must be satisfied.
          Dimension: (NUM_Z * NUM_PHI, )
      xstar: The interventional value of x in the objective.
      tmp_pre: Precomputed standard Guassian distributed values (for x).
          Dimension: (1, num_sample)
      xhats_pre: Precomputed samples following p(x | zi) for the zi in the
          Z grid (corresponding to the values in tmp_pre).
          Dimension: (NUM_Z, num_sample)
      sign: Either -1 or 1. If sign == 1, we are computing a lower bound.
          If sign == -1, we are computing an upper bound.

  Returns:
      a scalar estimate of the Lagrangian at the given eta and xstar
  """
  obj, _, psisum, _ = objective_rhs_psisum_constr(
      key, params, lmbda, tau, lhs, slack, xstar, tmp_pre, xhats_pre)
  return sign * obj + psisum


def init_params(key: np.ndarray) -> Params:
  """Initiliaze the optimization parameters."""
  key, subkey = random.split(key)
  # init diagonal at 0, because it will be exponentiated
  L = 0.05 * np.tri(dim_theta + 1, k=-1)
  L *= random.normal(subkey, (dim_theta + 1, dim_theta + 1))
  corr = make_correlation_matrix(L)
  assert np.all(np.isclose(np.linalg.cholesky(corr),
                           make_cholesky_factor(L))), "not PSD"
  key, subkey = random.split(key)
  if response_type == "poly":
    mu = 0.001 * random.normal(subkey, (dim_theta,))
    #mu = 0.5 * random.normal(subkey, (dim_theta,))
    #print("this is mu " + str(mu))
    log_sigma =np.array([np.log(1. / (i + 1))
                          for i in range(dim_theta)])
    #mu = 25 * random.normal(subkey, (dim_theta,))
    #log_sigma = 200 * np.array([np.log(1. / (i + 1))
    #                      for i in range(dim_theta)])
  elif response_type == "gp":
    mu = np.ones(dim_theta) / dim_theta
    log_sigma = 0.5 * np.ones(dim_theta)
  else:
    mu = 0.01 * random.normal(subkey, (dim_theta,))
    log_sigma = 0.5 * np.ones(dim_theta)
  params = (L, mu, log_sigma)
  return params


lagrangian_value_and_grad = jit(value_and_grad(lagrangian, argnums=1))


def run_optim(key: np.ndarray,
              lhs: np.ndarray,
              tmp: np.ndarray,
              xhats: np.ndarray,
              tmp_c: np.ndarray,
              xhats_c: np.ndarray,
              xstar: float,
              bound: Text,
              slack: float,
              slack_abs: float,
              num_z: int
              ) -> Tuple[int, float, float, int, float, float]:
  """Run optimization (either lower or upper) for a single xstar."""
  # Init optim params
  # ---------------------------------------------------------------------------
  key, subkey = random.split(key)
  params = init_params(subkey)

  tau = tau_init
  fin_tau = np.minimum(tau_factor**num_rounds * tau, tau_max)

  # Set constraint approach and slacks
  # ---------------------------------------------------------------------------
  slack_init = slack
  slack_abs_init = slack_abs
  slack = slack * np.ones(num_z * 2)
  lmbda = np.zeros(num_z * 2)

  slack *= np.abs(lhs.ravel())
  slack = np.maximum(slack_abs, slack)

  # Setup optimizer
  # ---------------------------------------------------------------------------
  step_size = optim.inverse_time_decay(
    lr, decay_steps, decay_rate, staircase)
  init_fun, update_fun, get_params = optim.sgd(step_size)
  state = init_fun(params)

  # Setup result dict
  # ---------------------------------------------------------------------------

  results = {
    "mu": [],
    "sigma": [],
    "cholesky_factor": [],
    "tau": [],
    "lambda": [],
    "objective": [],
    "constraint_term": [],
    "rhs": []
  }
  if plot_intermediate:
    results["grad_norms"] = []
    results["lagrangian"] = []

  sign = 1 if bound == "lower" else -1

  # ===========================================================================
  # OPTIMIZATION LOOP
  # ===========================================================================
  # One-time logging before first step
  # ---------------------------------------------------------------------------
  key, subkey = random.split(key)
  obj, rhs, psisum, constr = objective_rhs_psisum_constr(
    subkey, get_params(state), lmbda, tau, lhs, slack, xstar, tmp_c, xhats_c)

  results["objective"].append(obj)
  results["constraint_term"].append(psisum)
  results["rhs"].append(rhs)

  tril_idx = np.tril_indices(dim_theta + 1)
  count = 0
  for _ in tqdm(range(num_rounds)):
    # log current parameters
    # -------------------------------------------------------------------------
    results["lambda"].append(lmbda)
    results["tau"].append(tau)
    cur_L, cur_mu, cur_logsigma = get_params(state)
    cur_chol = make_cholesky_factor(cur_L)[tril_idx].ravel()[1:]
    results["mu"].append(cur_mu)
    results["sigma"].append(np.exp(cur_logsigma))
    results["cholesky_factor"].append(cur_chol)

    subkeys = random.split(key, num=opt_steps + 1)
    key = subkeys[0]
    # inner optimization for subproblem
    # -------------------------------------------------------------------------
    for j in range(opt_steps):
      v, grads = lagrangian_value_and_grad(
        subkeys[j + 1], get_params(state), lmbda, tau, lhs, slack, xstar,
        tmp, xhats, sign)
      state = update_fun(count, grads, state)
      count += 1
      if plot_intermediate:
        results["lagrangian"].append(v)
        results["grad_norms"].append([np.linalg.norm(grad) for grad in grads])

    # post inner optimization logging
    # -------------------------------------------------------------------------
    key, subkey = random.split(key)
    obj, rhs, psisum, constr = objective_rhs_psisum_constr(
      subkey, get_params(state), lmbda, tau, lhs, slack, xstar, tmp_c, xhats_c)
    results["objective"].append(obj)
    results["constraint_term"].append(psisum)
    results["rhs"].append(rhs)

    # update lambda, tau
    # -------------------------------------------------------------------------
    lmbda = update_lambda(constr, lmbda, tau)
    tau = np.minimum(tau * tau_factor, tau_max)

  # Convert and store results
  # ---------------------------------------------------------------------------
  results = {k: np.array(v) for k, v in results.items()}

  L, mu, log_sigma = get_params(state)
  results["final_L"] = L
  results["final_mu"] = mu
  results["final_log_sigma"] = log_sigma
  results["lhs"] = lhs

  # Compute last valid and last satisfied
  # ---------------------------------------------------------------------------
  maxabsdiff = np.array([np.max(np.abs(lhs - r)) for r in results["rhs"]])
  fin_i = np.sum(~np.isnan(results["objective"])) - 1
  fin_obj = results["objective"][fin_i]
  fin_maxabsdiff = maxabsdiff[fin_i]

  sat_i = [np.all((np.abs((lhs - r) / lhs) < slack_init) |
                  (np.abs(lhs - r) < slack_abs_init))
           for r in results["rhs"]]
  sat_i = np.where(sat_i)[0]

  if len(sat_i) > 0:
    sat_i = sat_i[-1]
    sat_obj = results["objective"][sat_i]
    sat_maxabsdiff = maxabsdiff[sat_i]
  else:
    sat_i = -1
    sat_obj, sat_maxabsdiff = np.nan, np.nan

  return fin_i, fin_obj, fin_maxabsdiff, sat_i, sat_obj, sat_maxabsdiff


def fit_bounds(z: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        xstar_grid: np.ndarray,
        seed: int=42,
        num_z: int=0,
        response_type: str="poly",
        dim_theta: int=2,
        bs: int=1024,
        bs_constr: int=4096,
               slack: float=0.2,
               slack_abs: float=0.2):
  """ general fit function """
  key = random.PRNGKey(seed)
  z_grid, bin_ids = utils_giv.make_zgrid_and_binids(z, num_z)

  basis_predict = None
  global response
  if response_type == "poly":
    @jit
    def response_poly(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
      """The response function."""
      return np.polyval(theta, x)

    response = response_poly
  elif response_type == "gp":
    basis_predict = utils_giv.get_gp_prediction(x, y, n_samples=dim_theta)

    @jit
    def response_gp(theta: np.ndarray, _x: np.ndarray) -> np.ndarray:
      _x = np.atleast_1d(_x)
      if _x.ndim == 1:
        # (n,) <- (1, k) @ (k, n)
        return (basis_predict(_x) @ theta).squeeze()
      else:
        # (n_constr, n) <- (n_constr, n, k) @ (k, n)
        return np.einsum('ijk,kj->ij', basis_predict(_x), theta)

    response = response_gp
  elif response_type == "mlp":
    key, subkey = random.split(key)
    basis_predict = utils_giv.fit_mlp(subkey, x[:, np.newaxis], y,
                                  n_samples=dim_theta)

    @jit
    def response_mlp(theta: np.ndarray, _x: np.ndarray) -> np.ndarray:
      _x = np.atleast_2d(_x)
      if _x.shape[0] == 1:
        # (n,) <- (1, k) @ (k, n)
        return (basis_predict(_x) @ theta).squeeze()
      else:
        # (n_constr, n) <- (n_constr, n, k) @ (k, n)
        return np.einsum('ijk,kj->ij', basis_predict(_x[:, :, None]), theta)

    response = response_mlp
  else:
    raise NotImplementedError(f"Unknown response_type {response_type}.")
  print(dim_theta)
  print(response)
  # make LHS constraints
  lhs = make_constraint_lhs(y, bin_ids, z_grid)
  num_z=2
  tmp, xhats = utils_giv.get_x_samples(x, bin_ids, num_z, bs)
  tmp_c, xhats_c = utils_giv.get_x_samples(x, bin_ids, num_z, bs_constr)
  num_xstar = len(xstar_grid)
  final = {
    "indices": np.zeros((num_xstar, 2), dtype=np.int32),
    "objective": np.zeros((num_xstar, 2)),
    "maxabsdiff": np.zeros((num_xstar, 2)),
  }
  satis = {
    "indices": np.zeros((num_xstar, 2), dtype=np.int32),
    "objective": np.zeros((num_xstar, 2)),
    "maxabsdiff": np.zeros((num_xstar, 2)),
  }
  for i, xstar in enumerate(xstar_grid):
    for j, bound in enumerate(["lower", "upper"]):
      logging.info(f"Run xstar={xstar}, bound={bound}...")
      vis = "=" * 10
      logging.info(f"{vis} {i * 2 + j + 1}/{2 * num_xstar} {vis}")
      fin_i, fin_obj, fin_diff, sat_i, sat_obj, sat_diff = run_optim(
        key,
        lhs,
        tmp,
        xhats,
        tmp_c,
        xhats_c,
        xstar,
        bound,
      slack,
      slack_abs,
      num_z)
      final["indices"] = index_update(final["indices"], (i, j), fin_i)
      final["objective"] = index_update(final["objective"], (i, j), fin_obj)
      final["maxabsdiff"] = index_update(final["maxabsdiff"], (i, j), fin_diff)
      satis["indices"] = index_update(satis["indices"], (i, j), sat_i)
      satis["objective"] = index_update(satis["objective"], (i, j), sat_obj)
      satis["maxabsdiff"] = index_update(satis["maxabsdiff"], (i, j), sat_diff)
      print(final)
      print(satis)
  return satis, final