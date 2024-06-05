"""Tools for Hyperbolic PDE's"""
from dataclasses import dataclass
import numpy as np


def intl_cond_1D(x):
    """1D sample condition containing discontinuous and smooth sections"""
    return np.where((2 <= x) & (x <= 4),
                    1,
                    np.where((6 <= x) & (x <= 8),
                             (1 - np.cos(np.pi * x)) / 2,
                             0))


class HyperbolicSystem1D:
    """Hyperbolic system of the form qt + A @ qx = 0"""

    def __init__(self, A, x, t):
        self.A = A
        self.x = x
        self.t = t
        self.eigenvalues, self.eigenvectors = np.linalg.eig(A)

        self.R_r = np.array(self.eigenvectors)  # right matrix of eigenvectors
        self.R_l = np.linalg.inv(self.R_r)
        Lambda_test = self.R_l @ A @ self.R_r  # verification
        self.Lambda = np.diag(self.eigenvalues)  # use to discard float error

        self._eigenmotions = None
        self.method_str = None

    def eigenmotions(self, q0, method_class):
        """
        :param q0: array of initial states
        :param method_class: class from advection_solvers; Upwind or subclass
        :return: list of solved eigenmotions as Upwind object
        """
        w0 = self.R_l @ q0
        _eigenmotions = []
        for i, eigenvalue in enumerate(self.eigenvalues):
            _eigenmotions.append(method_class(w0[i], eigenvalue, self.x, self.t))

        self._eigenmotions = _eigenmotions
        return _eigenmotions

    def solve(self, q0, method_class):
        """
        :param q0: array of initial states for t=0
        :param method_class: class from advection_solvers; Upwind or subclass
        :return: solved Q for all time steps
        """
        eigenmotions = self.eigenmotions(q0, method_class)
        w = np.array([eigenmotion.Q for eigenmotion in eigenmotions])

        Q = np.zeros_like(w)
        for i in range(np.shape(Q)[1]):
            Q[:, i, :] = self.R_r @ w[:, i, :]

        self.method_str = str(method_class)
        return Q

    @property
    def name(self):
        return str(self._eigenmotions[0])
