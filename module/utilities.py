import time
from matplotlib import pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from module.trajectory import Trajectory


def run_steplength_test(
    func,
    x0=np.array([[1.05], [0.50]]),
    T_values=[50, 100, 200],
    dt_values=[1, 0.1, 0.01, 0.001],
):
    fig, ax = plt.subplots(len(T_values), 1, figsize=(10, 10))
    for i, tf in enumerate(T_values):
        for dt in dt_values:
            traj = Trajectory(x0=x0, T=(0, tf), dt=dt, Np=1)
            X = traj(func)
            ax[i].plot(X[:, 0, :], X[:, 1, :], label=f"dt = {dt}")
        ax[i].set_title(f"Trajectory for T = {tf}")
        ax[i].set_xlabel("X")
        ax[i].set_ylabel("Y")
        ax[i].grid(True)
        ax[i].legend()

    plt.tight_layout()
    plt.show()


def run_timing_test(func, T=(0, 10), dt=0.01, Np_values=[1, 100, 1000, 10000, 10000]):
    times = []

    for Np in Np_values:
        x0 = np.random.rand(2, Np)
        start_time = time.time()
        traj = Trajectory(x0, T=T, dt=dt, Np=Np)
        traj(func)
        end_time = time.time()
        times.append(end_time - start_time)

    # Calculate linear times for comparison outside the loop
    linear_times = [Np * (times[0] / Np_values[0]) for Np in Np_values]

    plot_times(Np_values, times, linear_times)
    return Np, times, linear_times


def plot_times(Np_values, times, linear_times):
    plt.plot(Np_values, times, marker="o", label="Actual Time")
    plt.plot(Np_values, linear_times, marker="o", linestyle="--", label="Linear Time")
    plt.xlabel("Number of Particles (Np)")
    plt.ylabel("Run Time (s)")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Run Time vs. Number of Particles")
    plt.grid(True)
    plt.legend()
    plt.show()


def velocity(X, t):
    A, eps, w = 0.10, 0.25, 1  # Initial constants
    a = eps * np.sin(w * t)  # Equation 4a
    b = 1 - 2 * eps * np.sin(w * t)

    x, y = X  # Assuming X is a 2-element array or has shape (2, N)

    f = a * x**2 + b * x  # Equation 4b
    dx = -np.pi * A * np.sin(np.pi * f) * np.cos(np.pi * y)  # Equation 3
    dy = np.pi * A * np.cos(np.pi * f) * np.sin(np.pi * y) * (2 * a * x + b)
    return np.array([dx, dy])


def plot_map(
    f,
    X,
    V,
    L=None,
    figsize=(15, 12),
    extent=(7, 11, 63.5, 65),
    s=2,
    scatter_color=("blue", "red"),
    land_color="purple",
    plot_linewidth=0.1,
    plot_color="lightblue",
    alpha=0.1,
    stream_linewidth=1,
    stream_color="black",
    density=1,
):
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.NorthPolarStereo())

    # Adding land feature
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "physical", "land", "10m", edgecolor="face", facecolor="#cccccc"
        )
    )

    # Projection object for converting particle positions
    projection_params = {
        "central_latitude": 90,
        "central_longitude": f.dataset.projection_stere.straight_vertical_longitude_from_pole,
        "false_easting": f.dataset.projection_stere.false_easting,
        "false_northing": f.dataset.projection_stere.false_northing,
        "true_scale_latitude": f.dataset.projection_stere.standard_parallel,
    }
    npstere = ccrs.Stereographic(**projection_params)

    ax.plot(
        X[:, 0, :],
        X[:, 1, :],
        transform=npstere,
        color=plot_color,
        linewidth=plot_linewidth,
        alpha=alpha,
    )

    ax.streamplot(
        X[:, 0, :],
        X[:, 1, :],
        V[:, 0, :],
        V[:, 1, :],
        transform=npstere,
        color=stream_color,
        linewidth=stream_linewidth,
        density=density,
        arrowstyle="->",
        arrowsize=1,
    )
    # Plot initial and final positions
    ax.scatter(
        X[0, 0, :],
        X[0, 1, :],
        transform=npstere,
        s=s,
        label="Initial positions",
        color=scatter_color[0],
    )
    ax.scatter(
        X[-1, 0, :],
        X[-1, 1, :],
        transform=npstere,
        s=s,
        label="Final positions",
        color=scatter_color[1],
    )
    if L is not None:
        ax.scatter(
            L[0, :],
            L[1, :],
            transform=npstere,
            s=s,
            label="Final positions",
            color=land_color,
        )
    # Setting the map extent and adding gridlines
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.gridlines(
        draw_labels=True,
        dms=True,
        x_inline=False,
        y_inline=False,
        alpha=0.5,
        color="k",
        lw=0.5,
    )

    ax.legend()
    plt.tight_layout()
    plt.show()


def calculate_extent(X):
    min_lon, max_lon = np.min(X[:, 0, :]), np.max(X[:, 0, :])
    min_lat, max_lat = np.min(X[:, 1, :]), np.max(X[:, 1, :])
    extent = (min_lon - 1, max_lon + 1, min_lat - 1, max_lat + 1)
    return extent


def test_calculate_extent():
    X = np.zeros((2, 10))
    X[0, :] = np.random.normal(loc=790000, scale=10000, size=10)
    X[1, :] = np.random.normal(loc=490000, scale=10000, size=10)
    extent = calculate_extent(X)
    assert len(extent) == 4
    assert extent[0] < extent[1]
    assert extent[2] < extent[3]
    print("All tests passed!")
