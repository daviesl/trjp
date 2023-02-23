import matplotlib.pyplot as plt
from KDEpy import FFTKDE
import numpy as np
import scipy.stats as stats
from matplotlib.pyplot import imshow


def kde_1D(
    ax,
    data,
    cmap="jet",
    alpha=0.7,
    bw=0.25,
    color="b",
    fillcolor="lightblue",
    plotline=True,
):
    x, y = FFTKDE(kernel="gaussian", bw=bw).fit(data).evaluate()
    if plotline:
        ax.plot(x, y, color)
    ax.fill_between(x=x, y1=y, color=fillcolor)


def kde_joint(
    ax,
    data,
    cmap="jet",
    alpha=0.7,
    bw=0.25,
    N=32,
    maxz_scale=2,
    plotlines=False,
    n_grid_points=1024,
):
    # bw = (data.shape[0] * (2 + 2) / 4.)**(-1. / (2 + 4))
    grid_points = n_grid_points  # 2**10  # Grid points in each dimension
    # N = 8 # Number of contours
    kde = FFTKDE(bw=bw)
    grid, points = kde.fit(data).evaluate(grid_points)
    # The grid is of shape (obs, dims), points are of shape (obs, 1)
    x, y = np.unique(grid[:, 0]), np.unique(grid[:, 1])
    z = points.reshape(grid_points, grid_points).T
    # Plot the kernel density estimate
    maxz = z.max()
    minz = z.min()
    # Nlog = np.log10(np.logspace(minz,maxz,N))
    Nlog = np.geomspace(1e-15 * maxz, maxz_scale * maxz, 3 * N)
    #print("nlog", Nlog)
    if plotlines:
        ax.contour(x, y, z, Nlog, linewidths=0.5, colors="k", alpha=0.5)
    ax.contourf(x, y, z, Nlog, cmap=cmap, alpha=alpha)


def plot_bivariates_scatter(data):
    # plt.clf()
    # Create 2D data of shape (obs, dims)
    # data = np.random.randn(2**4, 2)
    dim = data.shape[1]
    fig, axs = plt.subplots(nrows=dim, ncols=dim)
    for i in range(dim):
        for j in range(i):
            ax = axs[i, j]
            ax.scatter(data[:, i], data[:, j], s=1)
    plt.show()


def plot_bivariates(data):
    plt.clf()
    # Create 2D data of shape (obs, dims)
    # data = np.random.randn(2**4, 2)
    bw = (data.shape[0] * (2 + 2) / 4.0) ** (-1.0 / (2 + 4))
    bw = 0.25
    grid_points = 2**10  # Grid points in each dimension
    N = 7  # Number of contours
    dim = data.shape[1]
    fig, axs = plt.subplots(nrows=dim, ncols=dim)
    for i in range(dim):
        for j in range(i):
            # ax = fig.add_subplot(int(np.ceil(np.sqrt(num))), int(np.ceil(np.sqrt(num))), int(i))
            ax = axs[i, j]
            kde = FFTKDE(bw=bw)
            # grid, points = kde.fit(data[:,[i-1,i]]).evaluate(grid_points)
            grid, points = kde.fit(data[:, [i, j]]).evaluate(grid_points)

            # The grid is of shape (obs, dims), points are of shape (obs, 1)
            x, y = np.unique(grid[:, 0]), np.unique(grid[:, 1])
            z = points.reshape(grid_points, grid_points).T

            # Plot the kernel density estimate
            ax.contour(x, y, z, N, linewidths=0.5, colors="k", alpha=0.5)
            ax.contourf(x, y, z, N, cmap="jet", alpha=0.7)
