# -*- coding: utf-8 -*-
"""optimization.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1jEYomZAvjGC82QdoQ0NUCwUdXuwGFrf7
"""

import numpy as np
import scipy
from utils import get_line_search_tool
from collections import defaultdict
from scipy.optimize.linesearch import scalar_search_wolfe2
import oracles
import time
from scipy.sparse import diags
np.random.seed(11)


#from optimization_old import gradient_descent

def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False, display=False):
    """
    Gradien descent optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively.
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format and is up to a student and is not checked in any way.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        "success" or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2

    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = gradient_descent(oracle, np.zeros(5), line_search_options={'method': 'Armijo', 'c1': 1e-4})
    >> print('Found optimal point: {}'.format(x_opt))
       Found optimal point: [ 0.  1.  2.  3.  4.]
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    f_k = oracle.func(x_k)
    g_k = oracle.grad(x_k)

    g_k_norm = np.linalg.norm(g_k)
    g_norm_0 = np.copy(np.linalg.norm(g_k))

    alpha_new = None

    startTime =  time.time()

    for k in range(max_iter):
      if np.isinf(x_k).any() or np.isnan(x_k).any():
        return x_k, 'computational_error', history

      if trace:
        history['time'].append(time.time()-startTime)
        history['func'].append(np.copy(f_k))
        history['grad_norm'].append(np.copy(g_k_norm))
        if x_k.size <= 2:
          history['x'].append(np.copy(x_k))

      if (g_k_norm)**2 <= tolerance*(g_norm_0**2):
        break

      alpha_new = line_search_tool.line_search(oracle, x_k, -g_k, 2 * alpha_new if alpha_new else None)

      x_k = x_k - alpha_new*g_k

      f_k = oracle.func(x_k)
      g_k = oracle.grad(x_k)
      g_k_norm = np.linalg.norm(g_k)

    if (k+1 == max_iter):
      return x_k, 'iterations_exceeded', history

    return x_k, 'success', history

def conjugate_gradients(matvec, b, x_0, tolerance=1e-4, max_iter=None, trace=False, display=False):
    """
    Solves system Ax=b using Conjugate Gradients method.

    Parameters
    ----------
    matvec : function
        Implement matrix-vector product of matrix A and arbitrary vector x
    b : 1-dimensional np.array
        Vector b for the system.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
        Stop optimization procedure and return x_k when:
         ||Ax_k - b||_2 <= tolerance * ||b||_2
    max_iter : int, or None
        Maximum number of iterations. if max_iter=None, set max_iter to n, where n is
        the dimension of the space
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display:  bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['residual_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """

    history = defaultdict(list) if trace else None
    x_k = np.copy(x_0)
    # TODO: Implement Conjugate Gradients method.

    g_k = matvec(x_k) - b
    d_k = -g_k

    b_norm = scipy.linalg.norm(b)
    g_k_norm = scipy.linalg.norm(b)

    max_iter = x_k.size if not max_iter else max_iter

    startTime =  time.time()

    for iter in range(max_iter):

      if trace:
        history['time'].append(time.time()-startTime)
        history['residual_norm'].append(np.copy(g_k_norm))
        if x_k.size <= 2:
          history['x'].append(np.copy(x_k))

      if g_k_norm <= tolerance * b_norm:
          message = 'success'
          break

      Ad_k = matvec(d_k)

      alpha = np.dot(g_k.T, g_k) / np.dot(d_k.T, Ad_k)
      x_k = x_k + alpha * d_k

      g_k_previous = np.copy(g_k)
      g_k = g_k + alpha * Ad_k

      g_k_norm = np.linalg.norm(g_k)

      beta = np.dot(g_k.T , g_k) / np.dot(g_k_previous.T , g_k_previous)
      d_k = -g_k + beta * d_k

    if (iter+1 == max_iter):
      return x_k, 'iterations_exceeded', history

    return x_k, 'success', history


def lbfgs(oracle, x_0, tolerance=1e-4, max_iter=500, memory_size=10,
          line_search_options=None, display=False, trace=False):
    """
    Limited-memory Broyden–Fletcher–Goldfarb–Shanno's method for optimization.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func() and .grad() methods implemented for computing
        function value and its gradient respectively.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    memory_size : int
        The length of directions history in L-BFGS method.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0).astype(np.float64)

    # TODO: Implement L-BFGS method.
    # Use line_search_tool.line_search() for adaptive step size.

    g_k = oracle.grad(x_k)
    g_0_norm = np.linalg.norm(g_k)
    g_k_norm = np.linalg.norm(g_k)

    startTime =  time.time()

    def bfgs_multiply(v, H, gamma_0):
      if len(H) == 0:
        return gamma_0 * v
      s, y = H[-1]
      H = H[:-1]
      v_ = v - np.dot(np.dot(s, v) / np.dot(y, s) , y)
      z = bfgs_multiply(v_, H, gamma_0)
      return z + np.dot((np.dot(s, v) - np.dot(y, z)) / np.dot(y, s), s)

    def lbfgs_direction():
      if len(H) == 0:
          return -g_k
      s, y = H[-1]
      gamma_0 = np.dot(y, s) / np.dot(y, y)
      return bfgs_multiply(-g_k, H, gamma_0)

    H = []

    for iter in range(max_iter):
      if trace:
        history['time'].append(time.time()-startTime)
        history['func'].append(np.copy(oracle.func(x_k)))
        history['grad_norm'].append(np.copy(g_k_norm))
        if x_k.size <= 2:
          history['x'].append(np.copy(x_k))

      if g_k_norm ** 2 < tolerance * g_0_norm ** 2:
        message = 'success'
        break

      d = lbfgs_direction()
      alpha = line_search_tool.line_search(oracle, x_k, d)
      x_k_ = x_k + alpha * d
      g_k_ = oracle.grad(x_k_)

      H.append((x_k_ - x_k, g_k_ - g_k))

      if len(H) > memory_size:
          H = H[1:]

      x_k, g_k = x_k_, g_k_
      g_k_norm = np.linalg.norm(g_k)

    if (iter+1 == max_iter):
      return x_k, 'iterations_exceeded', history

    return x_k, 'success', history


def hessian_free_newton(oracle, x_0, tolerance=1e-4, max_iter=500,
                        line_search_options=None, display=False, trace=False):
    """
    Hessian Free method for optimization.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess_vec() methods implemented for computing
        function value, its gradient and matrix product of the Hessian times vector respectively.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0).astype(np.float64)

    # TODO: Implement hessian-free Newton's method.
    # Use line_search_tool.line_search() for adaptive step size.

    g_k = oracle.grad(x_k)
    g_0_norm = np.linalg.norm(g_k)
    g_k_norm = np.linalg.norm(g_k)
    d_k = -g_k.copy()
    startTime =  time.time()


    for iter in range(max_iter):

      if trace:
        history['time'].append(time.time()-startTime)
        history['func'].append(np.copy(oracle.func(x_k)))
        history['grad_norm'].append(np.copy(g_k_norm))
        if x_k.size <= 2:
          history['x'].append(np.copy(x_k))

      eta = min(0.5, np.sqrt(g_k_norm))

      d_k_old = d_k.copy()
      while True:
        d_k, _, _ = conjugate_gradients(lambda d: oracle.hess_vec(x_k, d), -g_k, d_k_old, eta*g_k_norm)
        if np.dot(d_k, g_k) < 0:
            break
        eta /= 10

      if g_k_norm ** 2 < tolerance * g_0_norm ** 2:
        message = 'success'
        break

      alpha_k = line_search_tool.line_search(oracle, x_k, d_k, 1.0)
      x_k += alpha_k * d_k

      g_k = oracle.grad(x_k)
      g_k_norm = scipy.linalg.norm(g_k)

    if (iter+1 == max_iter):
      return x_k, 'iterations_exceeded', history

    return x_k, 'success', history