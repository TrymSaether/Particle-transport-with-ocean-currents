import time
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np
from module.trajectory import Trajectory

def run_steplength_test(
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
    # Plot the trajectories
    fig, ax = plt.subplots(2, len(time_testvalues), figsize=(14, 10), dpi= 150, sharex=True, sharey=True)
    time_array = np.zeros((len(time_testvalues), len(stepsize_testvalues)))
    for i, tf in enumerate(time_testvalues):
        ax[1, i].remove()
        ax[0,i].cla()
        ax[0,i].plot(X0[0], X0[1], "bo", **kwargs) # Plot of Initial position
        for j, dt in enumerate(stepsize_testvalues):
            traj = Trajectory(X0=X0, time_interval=[0, tf], stepsize=dt)

            tik = time.time()
            X, Y = traj(interpolator) # Trajectory calculation with velocity interpolator function using Heuns method
            time_array[i, j] = time.time() - tik

            ax[0,i].plot(X, Y, label=f"dt = {dt}", **kwargs) # Trajectory at different timesteps
            ax[0,i].plot(X[-1], Y[-1], "ro", **kwargs) # Plot of final position
            ax[0,i].set_title(f"Trajectory at T = {tf}")
            ax[0,i].set_xlabel("x")
            ax[0,i].set_ylabel("y")
            ax[0,i].grid(True)

    # Plot the runtime vs. timestep
    bottom_ax = fig.add_subplot(212)
    bottom_ax.plot(stepsize_testvalues, time_array[0, :], label="T = 50")
    bottom_ax.plot(stepsize_testvalues, time_array[1, :], label="T = 100")
    bottom_ax.set_xscale("log")
    bottom_ax.set_title("Runtime vs. timestep")
    bottom_ax.set_xlabel("timestep")
    bottom_ax.set_ylabel("runtime")
    bottom_ax.grid()
    bottom_ax.legend()
    fig.suptitle("Steplength test")
    fig.legend(*ax[0, 0].get_legend_handles_labels(), loc="upper left")
    plt.tight_layout()
    plt.show()




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


def create_gif(trajectory, **args):
    X, Y = trajectory.get_XY()
    Np = trajectory.number_of_particles
    T = trajectory.time_interval[1]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.scatter(X[0, :], Y[0, :], s=0.5, color='darkblue')
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid()
    ax.set_title(f"Trajectories of $N_p = {Np}$ particles over T = {T}")
    def animate(i):
        ax.clear()
        ax.plot(X[:i, :], Y[:i, :], color='cornflowerblue', alpha=0.3, linewidth=0.1)
        ax.scatter(X[i, :], Y[i, :], color='green', s=0.5)
        if i == X.shape[0]:
            ax.scatter(X[-1, :], Y[-1, :], color='red', s=0.5)
        
        

    


    anim = FuncAnimation(fig, animate, frames=X.shape[0], interval=100)
    # Save gif
    anim.save("trajectories.gif", writer="pillow")
    plt.show()
