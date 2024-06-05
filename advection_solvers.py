"""
Methods for the 1D solution of the advection equation
Boundary condition is periodic, sptial and temporal grid may be arbitrary
"""
import numpy as np


class Upwind:
    """
    Implementation of flux limiter finite volume methods, commonality according to pg. 115 of Leveque

    Methods work for 1D advection equations; hyperbolic systems can be seperated into 1D advection eqs for compatability

    Default flux limiter is upwind, others implemented via subclasses overwriting phi(theta)
    """
    def __repr__(self):
        return 'Upwind method'

    def __str__(self):
        return repr(self)

    def __init__(self, Q_0, u, x, t):
        """
        :param Q_0: initial state on spatial domain, must match x
        :param u: wave speed
        :param x: spatial domain
        :param t: time domain
        """
        self.x = x
        self.t = t
        self.u = u

        self.Q = np.zeros([len(t), len(x)])
        self.Q[0] = Q_0

        dx = np.zeros_like(x)  # initialise
        diffx = np.diff(x)  # get differences between cells
        diffx = np.take(diffx, np.arange(len(diffx)+2)-1, mode='wrap')  # connect left and right
        for i in range(len(dx)):  # compute cell width
            dx[i] = 0.5 * diffx[i] + 0.5 * diffx[i+1]
        self.dx = dx

        self.dt = np.diff(t)
        self.v_arr = u * self.dt[:, np.newaxis] / self.dx[np.newaxis, :]

        for n, Q in enumerate(self.Q[:-1]):
            Q_next = np.zeros_like(Q)
            for i, q in enumerate(Q):
                Qi = np.take(Q, [i - 2, i - 1, i, i + 1, i+2], mode='wrap')
                v = self.v_arr[n, i]

                if self.u > 0.:
                    self.di = 0  # theta indexing wavespeed dependency
                    dQ = (- v * (Qi[2] - Qi[1])
                          - 0.5 * v * (1 - v) * (self.delta(2, 3, Qi) - self.delta(1, 2, Qi)))

                elif self.u < 0.:
                    self.di = 2  # theta indexing wavespeed dependency
                    dQ = (- v * (Qi[3] - Qi[2])
                          + 0.5 * v * (1 + v) * (self.delta(2, 3, Qi) - self.delta(1, 2, Qi)))

                else:
                    dQ = 0.

                Q_next[i] = Qi[2] + dQ

            self.Q[n+1] = Q_next

    def delta(self, idx1, idx2, Qloc):
        """
        Step limiter function, overwritten by subclasses.
        Adapted Leveque flux limiters to avoid singularities on sections of zero slope
        :param idx1: left index of boundary in local Q
        :param idx2: right index of boundary  in local Q
        :param Qloc: local Q array
        :return: step limiter
        """
        return 0.

    def theta(self, idx1, idx2, Qloc):
        """
        theta calculation for condition handling in high resulution flux limiters, incl error handling
        :param idx1: left index of boundary in local Q
        :param idx2: right index of boundary  in local Q
        :param Qloc: local Q array
        :return: theta
        """
        denom = Qloc[idx2] - Qloc[idx1]
        if denom == 0.:
            return 10.  # effectively infinite, avoids div 0 exception
        else:
            return (Qloc[idx2 - 1 + self.di] - Qloc[idx1 - 1 + self.di]) / denom

    @staticmethod
    def minmod(a, b):
        if abs(a) < abs(b) and a*b > 0.:
            return a
        elif abs(a) > abs(b) and a*b > 0.:
            return b
        else:  # if a * b <= 0.
            return 0


class LaxWendroff(Upwind):
    def __repr__(self):
        return 'Lax-Wendroff method'

    def delta(self, idx1, idx2, Qloc):
        """
        Step limiter function, overwritten by subclasses.
        Adapted Leveque flux limiters to avoid singularities on sections of zero slope
        :param idx1: left index of boundary in local Q
        :param idx2: right index of boundary  in local Q
        :param Qloc: local Q array
        :return: step limiter
        """
        return Qloc[idx2] - Qloc[idx1]

    def __init__(self, Q_0, u, x, t):
        super().__init__(Q_0, u, x, t)


class BeamWarming(Upwind):
    def __repr__(self):
        return 'Beam-Warming method'

    def delta(self, idx1, idx2, Qloc):
        """
        Step limiter function, overwritten by subclasses.
        Adapted Leveque flux limiters to avoid singularities on sections of zero slope
        :param idx1: left index of boundary in local Q
        :param idx2: right index of boundary  in local Q
        :param Qloc: local Q array
        :return: step limiter
        """
        return Qloc[idx2 - 1 + self.di] - Qloc[idx1 - 1 + self.di]

    def __init__(self, Q_0, u, x, t):
        super().__init__(Q_0, u, x, t)


class Fromm(Upwind):
    def __repr__(self):
        return 'Fromm method'

    def delta(self, idx1, idx2, Qloc):
        """
        Step limiter function, overwritten by subclasses.
        Adapted Leveque flux limiters to avoid singularities on sections of zero slope
        :param idx1: left index of boundary in local Q
        :param idx2: right index of boundary  in local Q
        :param Qloc: local Q array
        :return: step limiter
        """
        return 0.5 * ((Qloc[idx2] - Qloc[idx1]) + (Qloc[idx2 - 1 + self.di] - Qloc[idx1 - 1 + self.di]))

    def __init__(self, Q_0, u, x, t):
        super().__init__(Q_0, u, x, t)


class Minmod(Upwind):
    def __repr__(self):
        return 'Minmod method'

    def delta(self, idx1, idx2, Qloc):
        """
        Step limiter function, overwritten by subclasses.
        Adapted Leveque flux limiters to avoid singularities on sections of zero slope
        :param idx1: left index of boundary in local Q
        :param idx2: right index of boundary  in local Q
        :param Qloc: local Q array
        :return: flux limiter
        """
        _theta = self.theta(idx1, idx2, Qloc)
        phi = self.minmod(1, _theta)

        return phi * (Qloc[idx2] - Qloc[idx1])

    def __init__(self, Q_0, u, x, t):
        super().__init__(Q_0, u, x, t)


class Superbee(Upwind):
    def __repr__(self):
        return 'Superbee method'

    def delta(self, idx1, idx2, Qloc):
        """
        Step limiter function, overwritten by subclasses.
        Adapted Leveque flux limiters to avoid singularities on sections of zero slope
        :param idx1: left index of boundary in local Q
        :param idx2: right index of boundary  in local Q
        :param Qloc: local Q array
        :return: flux limiter
        """
        _theta = self.theta(idx1, idx2, Qloc)
        phi = max(0, min(1, 2 * _theta), min(2, _theta))

        return phi * (Qloc[idx2] - Qloc[idx1])

    def __init__(self, Q_0, u, x, t):
        super().__init__(Q_0, u, x, t)


class MC(Upwind):
    def __repr__(self):
        return 'MC method'

    def delta(self, idx1, idx2, Qloc):
        """
        Step limiter function, overwritten by subclasses.
        Adapted Leveque flux limiters to avoid singularities on sections of zero slope
        :param idx1: left index of boundary in local Q
        :param idx2: right index of boundary  in local Q
        :param Qloc: local Q array
        :return: flux limiter
        """
        _theta = self.theta(idx1, idx2, Qloc)
        phi = max(0, min((1 + _theta) / 2, 2, 2 * _theta))

        return phi * (Qloc[idx2] - Qloc[idx1])

    def __init__(self, Q_0, u, x, t):
        super().__init__(Q_0, u, x, t)


class VanLeer(Upwind):
    def __repr__(self):
        return 'van Leer method'

    def delta(self, idx1, idx2, Qloc):
        """
        Step limiter function, overwritten by subclasses.
        Adapted Leveque flux limiters to avoid singularities on sections of zero slope
        :param idx1: left index of boundary in local Q
        :param idx2: right index of boundary  in local Q
        :param Qloc: local Q array
        :return: flux limiter
        """
        denom = Qloc[idx2] - Qloc[idx1]
        if denom == 0.:
            return 2*denom  # or 0, depending on which side of the limit you approach from
        else:
            _theta = self.theta(idx1, idx2, Qloc)
            return (_theta + abs(_theta)) / (1 + abs(_theta)) * denom

    def __init__(self, Q_0, u, x, t):
        super().__init__(Q_0, u, x, t)

