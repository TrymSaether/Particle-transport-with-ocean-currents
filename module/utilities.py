import time
from matplotlib import pyplot as plt
import numpy as np
from module.trajectory import Trajectory

def run_steplength_test(
    func,
    ax,
    x0=np.array([[0], [0]]),
    T_values=[50, 100],
    dt_values=[1, 0.1, 0.01, 0.001],
    **kwargs,
):
    time_array = np.zeros((len(T_values),len(dt_values)))
    for i, tf in enumerate(T_values):
        for j, dt in enumerate(dt_values):
            traj = Trajectory(x0=x0, T=(0, tf), dt=dt)
            init_time = time.time()
            X = traj(func)
            diff_time = time.time() - init_time
            time_array[i,j] = diff_time
            ax[i].plot(X[:, 0, :], X[:, 1, :], label=f"dt = {dt}", **kwargs)
        ax[i].set_title(f"T = {tf}")
    return ax, time_array


def run_timing_test(func, T=[0, 10], Np_values=[1, 100, 1000, 10000, 10000], dt=0.01):
    plt.figure(figsize=(7.5, 5))
    times = []
    for Np in Np_values:
        x0 = np.random.rand(2, Np)
        start_time = time.time()
        traj = Trajectory(x0, T=T, dt=dt, Np=Np)
        traj(func)
        end_time = time.time()
        times.append(end_time - start_time)

    linear_times = [Np * (times[0] / Np_values[0]) for Np in Np_values]
    return times, linear_times
