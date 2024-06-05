from advection_solvers import Upwind, LaxWendroff, BeamWarming, Superbee, Fromm, Minmod, VanLeer
from hyperbolic_tools import intl_cond_1D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Generate a colormap
colormap = cm.twilight  # You can choose any colormap you like

# Normalize the colormap to the number of lines you have
num_colors = 10  # Number of distinct lines/colors you want
colors = colormap(np.linspace(0, 1, num_colors))

# Set the global color cycle
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)


if __name__ == "__main__":
    u = -1.
    n_cycl = 10
    P = (n_cycl-1) * 10. / abs(u)  # period, plotting

    dx = 0.05
    x = np.arange(0., 10., dx)

    dt = 0.025
    t = np.arange(0., P+dt, dt)

    Q_0 = intl_cond_1D(x)

    stablemethods = [Upwind, LaxWendroff, BeamWarming, Superbee, Fromm, Minmod, VanLeer]
    n_methods = len(stablemethods)
    n_cols = 2
    n_rows = 4

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 20))
    # Collect all handles and labels for the shared legend
    handles = []
    labels = []

    for idx, method_class in enumerate(stablemethods):
        method = method_class(Q_0, u, x, t)
        ints = np.linspace(0., len(t) - 1, n_cycl, dtype=int)

        ax = axes[idx // n_cols, idx % n_cols]
        for i in ints:
            line, = ax.plot(x, method.Q[i], label=f't={np.round(method.t[i], 4)}')
            handles.append(line)
            labels.append(f't={np.round(method.t[i], 4)}')

        # Set title in the top right corner
        ax.text(0.95, 0.95, f'{method}',
                transform=ax.transAxes, ha='right', va='top', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    # Hide any empty subplots if the number of methods is less than the grid size
    for idx in range(n_methods, n_rows * n_cols):
        fig.delaxes(axes.flatten()[idx])

    fig.suptitle(f'Advection methods for {n_cycl} periods of {10. / abs(u)}[-], u = {u}[-], dx = {dx}[-], dt = {dt}[-]')

    # Add a shared legend in the empty subplot position
    # Get unique handles and labels
    unique_labels = list(dict.fromkeys(labels))  # Preserve order and remove duplicates
    unique_handles = [handles[labels.index(label)] for label in unique_labels]

    # Create an axis for the legend in the empty spot
    legend_ax = fig.add_subplot(n_rows, n_cols, n_methods + 1)
    legend_ax.axis('off')  # Hide the axis

    # Add the legend to this axis
    legend_ax.legend(unique_handles, unique_labels, loc='center', ncol=4, fontsize=9, columnspacing=0.8,
                     handletextpad=0.5)

    # plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the suptitle and legend
    plt.show()

    livemethods = []
    for method_ in livemethods:
        method = method_(Q_0, u, x, t)

        for i, ti in enumerate(t):
            plt.plot(x, method.Q[i], label=ti)
        plt.legend()
        plt.show()
