from hyperbolic_tools import HyperbolicSystem1D, intl_cond_1D
from advection_solvers import Upwind, LaxWendroff, BeamWarming, Superbee, Fromm, Minmod, VanLeer
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

dx = 0.05
x = np.arange(0., 10., dx)

dt = 0.025
t = np.arange(0., 1.+dt, dt)

Q0 = np.array([np.ones_like(x), intl_cond_1D(x)])

A = np.array([
    [1, 0],
    [0, 1]
]
Q0_1D = np.array([np.ones_like(x)])  # Also works for 1D!

A_1D = np.array([
    [1]
])


hsys = HyperbolicSystem1D(A, x, t)
Q = hsys.solve(Q0, Upwind)
print(np.shape(Q0))
print(np.shape(Q))

plt.plot(x, Q[0, -1])
plt.show()
