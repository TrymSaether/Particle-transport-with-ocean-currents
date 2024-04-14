import time
from matplotlib import pyplot as plt
import numpy as np
from module.trajectory import Trajectory

def run_steplength_test(
    ax,
    interpolator,
    X0,
    time_testvalues=[50, 100],
    stepsize_testvalues=[1, 0.1, 0.01, 0.001],
    **kwargs,
):
    """
    Run a steplength test for trajectory calculations.

    Parameters:
    - ax (matplotlib.axes.Axes): The axes object to plot the trajectories on.
    - interpolator (callable): The interpolator function to use for trajectory calculations.
    - X0 (tuple): The initial position of the trajectory.
    - time_testvalues (list, optional): The time values to test for each trajectory. Default is [50, 100].
    - stepsize_testvalues (list, optional): The stepsize values to test for each trajectory. Default is [1, 0.1, 0.01, 0.001].
    - **kwargs: Additional keyword arguments to pass to the plot functions.

    Returns:
    - ax (matplotlib.axes.Axes): The modified axes object.
    - time_array (numpy.ndarray): An array containing the execution times for each trajectory.

    """
    time_array = np.zeros((len(time_testvalues), len(stepsize_testvalues)))
    for i, tf in enumerate(time_testvalues):
        ax[i].cla()
        ax[i].plot(X0[0], X0[1], "bo", **kwargs)
        for j, dt in enumerate(stepsize_testvalues):
            traj = Trajectory(X0=X0, time_interval=[0, tf], stepsize=dt)
            tik = time.time()
            X, Y = traj(interpolator)
            tok = time.time() - tik
            time_array[i, j] = tok
            ax[i].plot(X, Y, label=f"dt = {dt}", **kwargs)
            ax[i].plot(X[-1], Y[-1], "ro", **kwargs)
            ax[i].set_title(f"Trajectory at T = {tf}")
            ax[i].set_xlabel("x")
            ax[i].set_ylabel("y")
            ax[i].grid(True)
            ax[i].set_aspect("equal")
        ax[i].set_aspect("equal")
    return ax, time_array


def run_timing_test(interpolator, number_of_particles_testvalues=[1, 100, 1000, 10000, 10000], time_interval=[0, 10],  stepsize=0.01):
    """
    Runs a timing test for the given interpolator.

    Parameters:
    - interpolator: The interpolator to be tested.
    - number_of_particles_testvalues: A list of number of particles to be tested.
    - time_interval: The time interval for the trajectory.
    - stepsize: The step size for the trajectory.

    Returns:
    - times: A list of execution times for each test.
    - linear_times: A list of linearly scaled execution times for each test.
    """
    plt.figure(figsize=(7.5, 5))
    times = []
    for Np in number_of_particles_testvalues:
        x0 = np.random.rand(2, Np)
        start_time = time.time()
        traj = Trajectory(x0, time_interval=time_interval, stepsize=stepsize, number_of_particles=Np)
        traj(interpolator)
        end_time = time.time()
        times.append(end_time - start_time)

    linear_times = [Np * (times[0] / number_of_particles_testvalues[0]) for Np in number_of_particles_testvalues]
    return times, linear_times