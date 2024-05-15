import numpy as np
import scipy
from scipy.special import expit
from scipy.sparse import issparse


def lasso_duality_gap(x, Ax_b, ATAx_b, b, regcoef):
    """
    Estimates f(x) - f* via duality gap for
        f(x) := 0.5 * ||Ax - b||_2^2 + regcoef * ||x||_1.
    """
    # TODO: implement.
    norm = scipy.linalg.norm(ATAx_b, ord = np.inf)
    aux = min(1.0, regcoef / norm) if norm else 1.0
    mu = aux * Ax_b
    return np.dot(Ax_b, Ax_b) / 2 + regcoef * scipy.linalg.norm(x, ord = 1) + np.dot(mu, mu) / 2 + np.dot(b, mu)


class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """

    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')

    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')


class Lasso(BaseSmoothOracle):
    def __init__(self, A, b, regcoef, t):
      self.A = A
      self.b = b
      self.regcoef = regcoef
      self.AT = A.T
      self.ATA = self.AT.dot(A)
      self.matvec_Ax = lambda x: A.dot(x)
      self.matvec_ATx = lambda x: self.AT.dot(x)
      self.t = t

    def func(self, x):
        x, u = np.array_split(x, 2)
        return self.t * ((np.linalg.norm(self.matvec_Ax(x) - self.b) ** 2) / 2 + self.regcoef * np.sum(u)) - np.sum(np.log(u + x) + np.log(u - x))

    def grad(self, x):
        x, u = np.array_split(x, 2)
        grad_x = self.t * self.matvec_ATx(self.matvec_Ax(x) - self.b) -1.0 / (u + x) + 1.0 / (u - x)
        grad_u = self.t * self.regcoef * np.ones(len(u)) -1.0 / (u + x) - 1.0 / (u - x)
        return np.concatenate((grad_x, grad_u))

    def hess(self, x):
        x, u = np.array_split(x, 2)
        hess_xx =  self.t * self.ATA + np.diag(1.0 / ((u - x) ** 2) + 1.0 / ((u + x) ** 2))
        hess_uu = np.diag(1.0 / ((u + x) ** 2) + 1.0 / ((u - x) ** 2))
        hess_xu = np.diag(1.0 / ((u + x) ** 2) - 1.0 / ((u - x) ** 2))
        return np.concatenate((np.concatenate((hess_xx, hess_xu), axis=1), np.concatenate((hess_xu, hess_uu), axis=1)), axis=0)

    def func_directional(self, x, d, alpha):
        x_k, u_k  = np.array_split(x, 2)
        d_x, d_u  = np.array_split(d, 2)
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        x_k, u_k  = np.array_split(x, 2)
        d_x, d_u  = np.array_split(d, 2)
        return np.squeeze(self.grad(x + alpha * d).dot(d))