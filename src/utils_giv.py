from typing import Tuple, Callable, Sequence, Text, Dict, Union

from typing import Tuple, Text
import itertools
from typing import Iterable

from absl import logging
import jax.numpy as np
import numpy as onp
import jax.scipy as sp
from jax import jit, vmap, grad
from jax.experimental import stax, optimizers
from jax.nn import softmax
from jax.scipy.special import logsumexp
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, PairwiseKernel


ArrayTup = Tuple[np.ndarray, np.ndarray]


# =============================================================================
# CDF and inverse CDF of Gaussians
# =============================================================================
@jit
def std_normal_cdf(x: np.ndarray) -> np.ndarray:
  """CDF of the standard normal."""
  return 0.5 * (1. + sp.special.erf(x / np.sqrt(2.)))


@jit
def normal_cdf_inv(x: np.ndarray,
                   mu: np.ndarray,
                   log_sigma: np.ndarray) -> np.ndarray:
  """Inverse CDF of a Gaussian with given mean and log standard deviation."""
  num = x.shape[-1]
  sigma = np.repeat(np.exp(log_sigma)[:, None], num, axis=-1)
  mu = np.repeat(mu[:, None], num, axis=-1)
  xx = np.clip(2 * x - 1, -0.999999, 0.999999)
  return np.sqrt(2.) * sigma * sp.special.erfinv(xx) + mu


# =============================================================================
# Other utils for boundiv
# =============================================================================
def ecdf(vals: np.ndarray, num_points: int = None) -> ArrayTup:
  """Evaluate the empirical distribution function on fixed number of points."""
  if num_points is None:
    num_points = len(vals)
  cdf = np.linspace(0, 1, num_points)
  t = np.quantile(vals, cdf)
  return t, cdf


def cdf_inv(vals: np.ndarray,
            num_points: int = None) -> Callable[..., np.ndarray]:
  """Compute an interpolation function of the (empirical) inverse cdf."""
  t, cdf = ecdf(vals, num_points)
  return lambda x: onp.interp(x, cdf, t)


def get_cdf_invs(val: np.ndarray,
                 bin_ids: np.ndarray,
                 num_z: int) -> Sequence[Callable[..., np.ndarray]]:
  """Compute a list of interpolated inverse CDFs of val at each z in Z grid."""
  cdf_invs = []
  for i in range(num_z):
    print(i)
    cdf_invs.append(cdf_inv(val[tuple([bin_ids == i])]))
  return cdf_invs


def get_z_bin_assigment(z: np.ndarray, z_grid: np.ndarray) -> np.ndarray:
  """Assignment of values in z to the respective bin in z_grid."""
  bins = np.concatenate((np.array([-np.inf]),
                         z_grid[1:-1],
                         np.array([np.inf])))
  hist = onp.digitize(z, bins=bins, right=True) - 1
  return hist


def get_x_samples(x: np.ndarray,
                  bin_ids: np.ndarray,
                  num_z: int,
                  num_sample: int) -> ArrayTup:
  """Pre-compute samples from p(x | z^{(i)}) for each gridpoint zi."""
  x_cdf_invs = get_cdf_invs(x, bin_ids, num_z)
  tmp = np.linspace(0, 1, num_sample + 2)[1:-1]
  tmp0 = normal_cdf_inv(tmp, np.array([0]), np.array([0]))
  return tmp0, np.array([x_cdf_inv(tmp) for x_cdf_inv in x_cdf_invs])


def get_y_pre(y: np.ndarray,
              bin_ids: np.ndarray,
              num_z: int,
              num_points: int) -> np.ndarray:
  """Compute the grid of y points for constraint approach y."""
  y_cdf_invs = get_cdf_invs(y, bin_ids, num_z)
  grid = np.linspace(0, 1, num_points + 2)[1:-1]
  return np.array([y_cdf_inv(grid) for y_cdf_inv in y_cdf_invs])


def make_zgrid_and_binids(z: np.ndarray, num_z: int) -> ArrayTup:
  """Discretize instrument Z and assign all z points to corresponding bins."""
  if num_z <= 0:
    logging.info("Discrete instrument specified, checking for values.")
    z_grid = np.sort(onp.unique(z))
    if len(z_grid) > 50:
      logging.info("Found more than 50 unique values for z. This is not a "
                   "discrete instrument. Aborting!")
      raise RuntimeError("Discrete instrument specified, but not found.")
    logging.info(f"Found {len(z_grid)} unique values for discrete instrument.")
    bin_ids = - onp.ones_like(z)
    for i, zpoint in enumerate(z_grid):
      bin_ids[z == zpoint] = i
    if onp.any(bin_ids < 0):
      raise ValueError(f"Found negative value in bin_ids. "
                       "Couldn't discretize instrument.")
    bin_ids = np.array(bin_ids).astype(int)
  else:
    z_grid = ecdf(z, num_z + 1)[0]
    bin_ids = get_z_bin_assigment(z, z_grid)
    z_grid = (z_grid[:-1] + z_grid[1:]) / 2
  return z_grid, bin_ids

# =============================================================================
# Basis functions: neural basis functions and GPs
# =============================================================================


def interp_regular_1d(x: np.ndarray,
                      xmin: float,
                      xmax: float,
                      yp: np.ndarray) -> np.ndarray:
  """One-dimensional linear interpolation.

  Returns the one-dimensional piecewise linear interpolation of the data points
  (xp, yp) evaluated at x. We extrapolate with the constants xmin and xmax
  outside the range [xmin, xmax].

  Args:
    x: The x-coordinates at which to evaluate the interpolated values.
    xmin: The lower bound of the regular input x-coordinate grid.
    xmax: The upper bound of the regular input x-coordinate grid.
    yp: The y coordinates of the data points.

  Returns:
    y: The interpolated values, same shape as x.
  """
  ny = len(yp)
  fractional_idx = (x - xmin) / (xmax - xmin)
  x_idx_unclipped = fractional_idx * (ny - 1)
  x_idx = np.clip(x_idx_unclipped, 0, ny - 1)
  idx_below = np.floor(x_idx)
  idx_above = np.minimum(idx_below + 1, ny - 1)
  idx_below = np.maximum(idx_above - 1, 0)
  y_ref_below = yp[idx_below.astype(np.int32)]
  y_ref_above = yp[idx_above.astype(np.int32)]
  t = x_idx - idx_below
  y = t * y_ref_above + (1 - t) * y_ref_below
  return y


interp1d = jit(vmap(interp_regular_1d, in_axes=(None, None, None, 0)))


def get_gp_prediction(x: np.ndarray,
                      y: np.ndarray,
                      n_samples: int,
                      n_points: int = 100):
  """Fit a GP to observed P( Y | X ) and sample some functions from it.

  Args:
    x: x-values (features)
    y: y-values (targets/labels)
    n_samples: The number of GP samples to use as basis functions.
    n_points: The number of points to subsample form x and y to fit each GP.

  Returns:
    a function that takes as input an array either of shape (n,) or (k, n)
    and outputs:
      if input is 1D -> output: (n, n_samples)
      if input is 2D -> output: (k, n, n_samples)
  """
  kernel = PairwiseKernel(metric='poly') + RBF()
  gp = GaussianProcessRegressor(kernel=kernel,
                                alpha=0.4,
                                n_restarts_optimizer=0,
                                normalize_y=True)
  xmin = np.min(x)
  xmax = np.max(x)
  xx = np.linspace(xmin, xmax, n_points)
  y_samples = []
  rng = onp.random.RandomState(0)
  for i in range(n_samples):
    logging.info("Subsample 200 points and fit GP to P(Y|X)...")
    idx = rng.choice(len(x), 200, replace=False)
    gp.fit(x[idx, np.newaxis], y[idx])
    logging.info(f"Get a sample functions from the GP")
    y_samples.append(gp.sample_y(xx[:, np.newaxis], 1))
  y_samples = np.array(y_samples).squeeze()
  logging.info(f"Shape of samples: {y_samples.shape}")

  def predict(inputs: np.ndarray) -> np.ndarray:
    return interp1d(inputs, xmin, xmax, y_samples).T
  return jit(vmap(predict))


def fit_mlp(key: np.ndarray,
            x: np.ndarray,
            y: np.ndarray,
            n_samples: np.ndarray,
            epochs: int = 100,
            batch_size: int = 256,
            learning_rate: float = 0.001,
            layers: Iterable[int] = (64, 64),
            return_basis=True):
  """Fit a small MLP to data.

  Args:
    key: Key for randomness.
    x: x-values (features)
    y: y-values (targets/labels)
    n_samples: The number of neurons in the additional last hidden layer which
        is used as basis functions.
    epochs: Number of epochs to train for.
    batch_size: batch size in MLP training
    learning_rate: Initial learning rate for Adam optimizer.
    layers: The hidden layer sizes. To this list, two dense layers of size
        n_samples and 1 are added.
    return_basis: Whether to return a function that returns the activations of
        the last layer instead of simply the full MLP model itself.

  Returns:
    a function that takes as input an array either of shape (k, n)
    and outputs and array of shape (k, n, n_samples)
  """
  logging.info(f"Fit small mlp with {layers} neurons, batchsize {batch_size} "
               f"for {epochs} epochs with Adam and lr={learning_rate}")
  seq = []
  for i in layers:
    seq.append(stax.Dense(i))
    seq.append(stax.Relu)
  # the final hidden layer used as basis functions
  seq.append(stax.Dense(n_samples))
  seq.append(stax.Relu)
  seq.append(stax.Dense(1))
  init_fun, mlp = stax.serial(*seq)
  opt_init, opt_update, get_params = optimizers.adam(learning_rate)
  _, init_params = init_fun(key, (-1, x.shape[-1]))
  opt_state = opt_init(init_params)

  n_train = x.shape[0]
  num_complete_batches, leftover = divmod(n_train, batch_size)
  num_batches = num_complete_batches + bool(leftover)

  def data_stream():
    rng = onp.random.RandomState(0)
    while True:
      perm = rng.permutation(n_train)
      for j in range(num_batches):
        batch_idx = perm[j * batch_size:(j + 1) * batch_size]
        yield x[batch_idx], y[batch_idx]
  batches = data_stream()
  itercount = itertools.count()

  @jit
  def loss(_params, _batch):
    inputs, targets = _batch
    preds = mlp(_params, inputs).squeeze()
    return np.mean((preds - targets) ** 2)

  @jit
  def update(_i, _opt_state, _batch):
    _params = get_params(_opt_state)
    return opt_update(_i, grad(loss)(_params, _batch), _opt_state)

  for epoch in range(epochs):
    for _ in range(num_batches):
      opt_state = update(next(itercount), opt_state, next(batches))
    params = get_params(opt_state)
    train_loss = loss(params, (x, y))
    logging.info(f"Epoch: {epoch + 1} / {epochs}")
    logging.info(f"    Loss: {train_loss}")

  # Copy of the original MLP without the last layer
  if return_basis:
    _, out = stax.serial(*seq[:-1])

    def predict(inputs: np.ndarray):
      return out(params[:-1], inputs)
  else:
    def predict(inputs: np.ndarray):
      return mlp(params, inputs)

  return jit(vmap(predict))


# =============================================================================
# Train mixture density network
# =============================================================================
def fit_mdn(key: np.ndarray,
            x: np.ndarray,
            y: np.ndarray,
            n_hidden: int = 20,
            n_mixture: int = 5,
            learning_rate: float = 0.001,
            batch_size: int = 256,
            n_epochs: int = 300):
  """Fit a mixture density network to the data.

  Args:
    x: Input values.
    y: Output values.
    n_hidden: Number of hidden neurons in the (only) hidden layer.
    n_mixture: Number of mixture components.
    learning_rate: The (fixed) learning rate for fitting with SGD.
    batch_size: Training batch size.
    n_epochs: Number of epochs to train for.

  Returns:
    the fitted predictor (as a callable function: x -> y)
  """
  log_sqrt_2pi = onp.log(onp.sqrt(2.0 * onp.pi))

  init_fun, network = stax.serial(stax.Dense(n_hidden),
                                  stax.Tanh,
                                  stax.Dense(n_mixture * 3))

  _, params = init_fun(key, (batch_size, x.shape[1]))

  opt_init, opt_update, get_params = optimizers.adam(learning_rate)
  opt_state = opt_init(params)

  def lognormal(_y, mean, logstd):
    return - 0.5 * ((_y - mean) / np.exp(logstd)) ** 2 - logstd - log_sqrt_2pi

  def get_mdn_coef(output):
    """Extract MDN coefficients."""
    logmix, mean, logstd = output.split(3, axis=1)
    logmix = logmix - logsumexp(logmix, 1, keepdims=True)
    return logmix, mean, logstd

  def mdn_loss_func(logmix, mean, logstd, _y):
    """MDN loss function."""
    v = logmix + lognormal(_y, mean, logstd)
    v = logsumexp(v, axis=1)
    return - np.mean(v)

  def loss_fn(_params, batch):
    """ MDN Loss function for training loop. """
    inputs, targets = batch
    outputs = network(_params, inputs)
    logmix, mean, logstd = get_mdn_coef(outputs)
    return mdn_loss_func(logmix, mean, logstd, targets)

  @jit
  def update(step, _opt_state, batch):
    """ Compute the gradient for a batch and update the parameters."""
    _params = get_params(_opt_state)
    grads = grad(loss_fn)(_params, batch)
    _opt_state = opt_update(step, grads, _opt_state)
    return _opt_state

  n_train = x.shape[0]
  num_complete_batches, leftover = divmod(n_train, batch_size)
  num_batches = num_complete_batches + bool(leftover)

  def data_stream():
    rng = onp.random.RandomState(342)
    while True:
      perm = rng.permutation(n_train)
      for j in range(num_batches):
        batch_idx = perm[j * batch_size:(j + 1) * batch_size]
        yield x[batch_idx], y[batch_idx]
  batches = data_stream()
  itercount = itertools.count()

  for epoch in range(n_epochs):
    for _ in range(num_batches):
      opt_state = update(next(itercount), opt_state, next(batches))
    params = get_params(opt_state)
    train_loss = loss_fn(params, (x, y))
    if epoch % 10 == 0:
      logging.info(f"Epoch: {epoch + 1} / {n_epochs}")
      logging.info(f"    Loss: {train_loss}")

  def predict(_x: np.ndarray):
    """Predict new values from MDN."""
    logmix, mu_data, logstd = get_mdn_coef(network(params, _x))
    pi_data = softmax(logmix)
    sigma_data = np.exp(logstd)
    z = onp.random.gumbel(loc=0, scale=1, size=pi_data.shape)
    k = (onp.log(pi_data) + z).argmax(axis=1)
    indices = (onp.arange(_x.shape[0]), k)
    rn = onp.random.randn(_x.shape[0])
    sampled = rn * sigma_data[indices] + mu_data[indices]
    return sampled

  return predict