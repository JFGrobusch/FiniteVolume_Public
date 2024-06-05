from advection_solvers import Upwind, LaxWendroff, BeamWarming, Superbee, Fromm, Minmod, VanLeer
from hyperbolic_tools import HyperbolicSystem1D, intl_cond_1D
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
"""Equation Parameters"""
u = - 1.0
mu = 1
rho = 0.25

A_advection = np.array([[u]])
A_stress = np.array([[0,     - 1 / rho],
                     [- mu,  0]
                     ])

"""Grid Parameters"""
dt = 0.05  # temporal resolution
dx = 0.2  # STANDARD spatial resolution

x_const = np.arange(0., 10., dx)
x_var = np.concatenate([np.arange(0., 5., 2*dx), np.arange(5., 10., dx)])

"""Initial Conditions"""
Q0_const_advection = np.array([intl_cond_1D(x_const)])
Q0_const_stress = np.array([np.ones_like(x_const), intl_cond_1D(x_const)])
Q0_var_advection = np.array([intl_cond_1D(x_var)])

"""Plotting Settings"""
PLOT_Q4_ADVECTION, PLOT_Q4_STRESS, PLOT_Q5 = True, True, True
nQ4 = [0, 1, 2, 5]
methods = [Upwind, BeamWarming, Superbee]
linestyle_advection = [plt.get_cmap('cividis')(value) for value in np.linspace(0, 1, 4)]
linestyle_stress = [[plt.get_cmap('magma')(value) for value in np.linspace(0, 1, 4)],
                    [plt.get_cmap('plasma')(value) for value in np.linspace(0, 1, 4)]]  # Blue, Green
label_stress = ['σ', 'v']
nQ5 = [0, 1, 5]

title = False
grid_x = True
show = False

# A4 paper size in inches
A4_width_inches = 8.27  # 210 mm converted to inches
A4_height_inches = 11.69  # 297 mm converted to inches

# Calculate the desired width and height
width = A4_width_inches / 1.5
height = (2 / 3) * width

# rc('figure', figsize=(11.69,8.27))

if __name__ == "__main__":
    if PLOT_Q4_ADVECTION:

        n_cycl = max(nQ4)  # No. of cycles to plot
        P = n_cycl * 10. / abs(u)  # Computed period from given u
        t = np.arange(0., P+dt, dt)  # Get time array
        t_plt = P * np.array(nQ4) / max(nQ4)
        idx = [np.argmin(np.abs(t - tn)) for tn in t_plt]  # get indices for plotting

        x = x_const

        for i, method in enumerate(methods):
            # Create a figure with the desired dimensions
            plt.figure(figsize=(width, height))

            hsys = HyperbolicSystem1D(A_advection, x, t)  # Initialise system (redundant for advection)
            Q = hsys.solve(Q0_const_advection, method)  # Solve system

            if grid_x:
                midpoints = (x[:-1] + x[1:]) / 2
                pady = 0.1
                ymin, ymax = -0.4, 1.2
                plt.vlines(midpoints, ymin, ymax, color='#b0b0b0', linewidth=0.5)
                # plt.hlines([0], 0., 10., color='#b0b0b0', linewidth=0.75)
                plt.ylim([ymin, ymax])
                plt.xlim([0., 10.])

            for j, t_idx in enumerate(idx):
                plt.step(x, Q[0, t_idx], color=linestyle_advection[-j], label=f't = {t_plt[j]}', where='mid')

            if title:
                plt.title(f'{hsys.name} with dx, dt = {dx}, {dt}')
            plt.xlabel('x')
            plt.ylabel('q')
            plt.legend()
            plt.tight_layout()

            if show:
                plt.show()
            else:
                ftitle = f'plots_adv/{hsys.name} {dx}{dt}.pdf'
                plt.savefig(ftitle, transparent=True, format='pdf')
                plt.close()
                print('Saved as', ftitle)

    if PLOT_Q4_STRESS:
        n_cycl = max(nQ4)  # No. of cycles to plot
        P = n_cycl * 10. / abs(u)  # Computed period from given u
        t = np.arange(0., P + dt, dt)  # Get time array
        t_plt = P * np.array(nQ4) / max(nQ4)
        idx = [np.argmin(np.abs(t - tn)) for tn in t_plt]  # get indices for plotting

        x = x_const

        for i, method in enumerate(methods):
            for k, linestyle in enumerate(linestyle_stress):
                # Create a figure with the desired dimensions
                plt.figure(figsize=(width, height))

                hsys = HyperbolicSystem1D(A_stress, x, t)  # Initialise system (redundant for advection)
                Q = hsys.solve(Q0_const_stress, method)  # Solve system

                if grid_x:
                    midpoints = (x[:-1] + x[1:]) / 2
                    pady = 0.1
                    plt.vlines(midpoints, np.min(Q[k]) - pady, np.max(Q[k]) + pady, color='#b0b0b0', linewidth=0.5)
                    # plt.hlines([0], 0., 10., color='#b0b0b0', linewidth=0.75)
                    plt.ylim([np.min(Q[k]) - pady, np.max(Q[k]) + pady])
                    plt.xlim([0., 10.])

                for j, t_idx in enumerate(idx):
                    plt.step(x, Q[k, t_idx], color=linestyle[-j], label=f't = {t_plt[j]}', where='mid')

                if title:
                    plt.title(f'{hsys.name} with dx, dt = {dx}, {dt}')
                plt.legend()
                plt.xlabel('x')
                plt.ylabel(label_stress[k])

                if show:
                    plt.show()
                else:
                    ftitle = f'plots_stress/{hsys.name} {label_stress[k]} {dx}{dt}.pdf'
                    plt.savefig(ftitle, transparent=True, format='pdf')
                    plt.close()
                    print('Saved as', ftitle)

    if PLOT_Q5:

        n_cycl = max(nQ4)  # No. of cycles to plot
        P = n_cycl * 10. / abs(u)  # Computed period from given u
        t = np.arange(0., P+dt, dt)  # Get time array
        t_plt = P * np.array(nQ4) / max(nQ4)
        idx = [np.argmin(np.abs(t - tn)) for tn in t_plt]  # get indices for plotting

        x = x_var

        for i, method in enumerate(methods):
            # Create a figure with the desired dimensions
            plt.figure(figsize=(width, height))

            hsys = HyperbolicSystem1D(A_advection, x, t)  # Initialise system (redundant for advection)
            Q = hsys.solve(Q0_var_advection, method)  # Solve system

            if grid_x:
                midpoints = (x[:-1] + x[1:]) / 2
                pady = 0.1
                ymin, ymax = -0.4, 1.2
                plt.vlines(midpoints, ymin, ymax, color='#b0b0b0', linewidth=0.5)
                # plt.hlines([0], 0., 10., color='#b0b0b0', linewidth=0.75)
                plt.ylim([ymin, ymax])
                plt.xlim([0., 10.])

            for j, t_idx in enumerate(idx):
                plt.step(x, Q[0, t_idx], color=linestyle_advection[-j], label=f't = {t_plt[j]}', where='mid')

            if title:
                plt.title(f'{hsys.name} with dx, dt = {dx} or {2*dx}, {dt}')
            plt.xlabel('x')
            plt.ylabel('q')
            plt.legend()
            plt.tight_layout()

            if show:
                plt.show()
            else:
                ftitle = f'plots_adv_vardx/{hsys.name} var{dx}{dt}.pdf'
                plt.savefig(ftitle, transparent=True, format='pdf')
                plt.close()
                print('Saved as', ftitle)
