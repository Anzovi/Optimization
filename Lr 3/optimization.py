from collections import defaultdict
import numpy as np
from numpy.linalg import norm, solve
import time
import datetime
from utils import get_line_search_tool
import oracles
import scipy
from numpy.linalg import LinAlgError

def barrier_method_lasso(A, b, reg_coef, x_0, u_0, tolerance=1e-5,
                         tolerance_inner=1e-8, max_iter=100,
                         max_iter_inner=20, t_0=1, gamma=10,
                         c1=1e-4, lasso_duality_gap=None,
                         trace=False, display=False):
    """
    Log-barrier method for solving the problem:
        minimize    f(x, u) := 1/2 * ||Ax - b||_2^2 + reg_coef * \sum_i u_i
        subject to  -u_i <= x_i <= u_i.

    The method constructs the following barrier-approximation of the problem:
        phi_t(x, u) := t * f(x, u) - sum_i( log(u_i + x_i) + log(u_i - x_i) )
    and minimize it as unconstrained problem by Newton's method.

    In the outer loop `t` is increased and we have a sequence of approximations
        { phi_t(x, u) } and solutions { (x_t, u_t)^{*} } which converges in `t`
    to the solution of the original problem.

    Parameters
    ----------
    A : np.array
        Feature matrix for the regression problem.
    b : np.array
        Given vector of responses.
    reg_coef : float
        Regularization coefficient.
    x_0 : np.array
        Starting value for x in optimization algorithm.
    u_0 : np.array
        Starting value for u in optimization algorithm.
    tolerance : float
        Epsilon value for the outer loop stopping criterion:
        Stop the outer loop (which iterates over `k`) when
            `duality_gap(x_k) <= tolerance`
    tolerance_inner : float
        Epsilon value for the inner loop stopping criterion.
        Stop the inner loop (which iterates over `l`) when
            `|| \nabla phi_t(x_k^l) ||_2^2 <= tolerance_inner * \| \nabla \phi_t(x_k) \|_2^2 `
    max_iter : int
        Maximum number of iterations for interior point method.
    max_iter_inner : int
        Maximum number of iterations for inner Newton's method.
    t_0 : float
        Starting value for `t`.
    gamma : float
        Multiplier for changing `t` during the iterations:
        t_{k + 1} = gamma * t_k.
    c1 : float
        Armijo's constant for line search in Newton's method.
    lasso_duality_gap : callable object or None.
        If calable the signature is lasso_duality_gap(x, Ax_b, ATAx_b, b, regcoef)
        Returns duality gap value for esimating the progress of method.
    trace : bool
        If True, the progress information is appended into history dictionary
        during training. Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.

    Returns
    -------
    (x_star, u_star) : tuple of np.array
        The point found by the optimization procedure.
    message : string
        "success" or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['func'] : list of function values f(x_k) on every **outer** iteration of the algorithm
            - history['duality_gap'] : list of duality gaps
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    # TODO: implement.

    history = defaultdict(list) if trace else None
    message = None

    oracle = oracles.Lasso(A, b, reg_coef, t_0)

    lasso_duality_gap = oracles.lasso_duality_gap

    AT = A.T
    matvec_Ax = lambda x: A.dot(x)
    duality_gap_func = lambda x: lasso_duality_gap(x, matvec_Ax(x) - b, np.dot(AT, matvec_Ax(x) - b), b, reg_coef)

    x_k, u_k = np.copy(x_0), np.copy(u_0)
    x_size = len(x_k)

    duality_gap_k = duality_gap_func(x_k)

    t_k = np.float64(t_0)
    startTime =  time.time()

    for k in range(max_iter):
      if trace:
        history['func'].append( (matvec_Ax(x_k) / 2) - b + reg_coef * scipy.linalg.norm(x_k, ord = 1))
        history['time'].append(time.time() - startTime)
        history['duality_gap'].append(duality_gap_k)
        if x_size <= 2:
            history['x'].append(np.copy(x_k))

      if duality_gap_k < tolerance:
        break

      oracle.t = t_k

      x_k, u_k, message_newton, _ = newton(oracle, x_k, u_k, max_iter=max_iter_inner, tolerance=tolerance_inner, line_search_options={'c1' : c1})

      if message_newton == 'computational_error':
          message = message_newton
          break

      t_k *= gamma
      duality_gap_k = duality_gap_func(x_k)

    if (k+1 == max_iter):
      return x_k, u_k, 'iterations_exceeded', history

    return x_k, u_k, 'success', history


def newton(oracle, x_0, u_0, tolerance=1e-5, max_iter=100,
           line_search_options=None, trace=False, display=False):

    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    u_k = np.copy(u_0)

    x_size = len(x_k)
    x = np.concatenate([x_k, u_k])

    alpha_new = None

    startTime =  time.time()

    func_k = oracle.func(x)
    grad_k = oracle.grad(x)
    hess_k = oracle.hess(x)

    startGrad = np.linalg.norm(grad_k).copy()
    grad_k_norm = np.linalg.norm(grad_k)


    def first_run(x, d, x_size):
      x_k, u_k = np.array_split(x, 2)
      grad_x, grad_u = np.array_split(d, 2)
      alpha = [1.0]
      theta = 0.99
      for i in range(x_size):
        if grad_x[i] > grad_u[i]:
          alpha.append(theta * (u_k[i] - x_k[i]) / (grad_x[i] - grad_u[i]))
        if grad_x[i] < -grad_u[i]:
          alpha.append(theta * (u_k[i] + x_k[i]) / (-grad_x[i] - grad_u[i]))
      return min(alpha)

    for k in range(max_iter):

      if x.dtype != float or np.isinf(x).any() or np.isnan(x).any():
        return (x_k, u_k), 'computational_error'

      if trace:
        history['time'].append(time.time()-startTime)
        history['func'].append(np.copy(func_k))
        history['grad_norm'].append(np.copy(grad_k_norm))
        if x_k.size <= 2:
          history['x'].append(np.copy(x))

      if ((grad_k_norm)**2 <= tolerance*(startGrad**2)).all():
        break

      if isinstance(hess_k, scipy.sparse.spmatrix):
            hess_k = hess_k.toarray()
      if hess_k.dtype != float or np.isinf(hess_k).any() or np.isnan(hess_k).any():
        return (x_k, u_k), 'computational_error'

      try:
          #cho_solve(cho_factor(Hess), np.eye(np.shape(Hess)[0])) -> Hess^-1; => Hess^-1 * (-grad_k)
          d_k = scipy.linalg.cho_solve(scipy.linalg.cho_factor(hess_k), -grad_k)
      except LinAlgError:
          return x_k, u_k, 'newton_direction_error', history

      alpha_new = line_search_tool.line_search(oracle, x, d_k, previous_alpha=first_run(x, d_k, x_size))

      x = x + alpha_new * d_k

      func_k = oracle.func(x)
      grad_k = oracle.grad(x)
      hess_k = oracle.hess(x)
      grad_k_norm = np.linalg.norm(grad_k)

    x_k, u_k = np.array_split(x, 2)

    if (k+1 == max_iter):
      return x_k, u_k, 'iterations_exceeded', history

    return x_k, u_k, 'success', history