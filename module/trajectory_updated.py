import numpy as np
from matplotlib import pyplot as plt
import time
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from interpolator import Interpolator
import xarray as xr


class Trajectory:
    def __init__(self, x0=None, T=(0, 10), dt=0.01, Np=100):
        self.dt = dt
        self.T = T
        self.Np = Np
        if x0 is not None:
            self.x0 = x0
        else:
            self.x0 = np.zeros((2, Np))  # Default initialization

    @classmethod
    def from_analytical(cls, Np, T, dt):
        instance = cls(T=T, dt=dt, Np=Np)
        instance.initialize_particles(mode="grid", size=0.1)
        return instance

    def __call__(self, func, method="heun"):
        return self.solve(func, method)

    def __repr__(self):
        return f"Trajectory(x0={self.x0.shape}, T={self.T}, dt={self.dt}, Np={self.Np})"

    def initialize_particles(
        self, mode="random", loc=(790000, 490000), scale=10000, size=0.1
    ):
        if mode == "random":
            self.x0[0, :] = np.random.normal(loc=loc[0], scale=scale, size=self.Np)
            self.x0[1, :] = np.random.normal(loc=loc[1], scale=scale, size=self.Np)
        elif mode == "linear":
            self.x0[0, :] = np.linspace(1e-3, size, self.Np)
            self.x0[1, :] = np.linspace(1e-3, size, self.Np)
        elif mode == "grid":
            sqrt_Np = int(np.sqrt(self.Np))
            if sqrt_Np**2 != self.Np:
                raise ValueError("Np must be a perfect square for grid mode.")
            grid_x, grid_y = np.meshgrid(
                np.linspace(1e-3, size, sqrt_Np), np.linspace(1e-3, size, sqrt_Np)
            )
            self.x0 = np.vstack((grid_x.flatten(), grid_y.flatten()))
        else:
            raise ValueError("Invalid initialization mode")

    def heun_method(self, func, X0):
        """
        Implementation of Heun's method for solving ODEs, with land-hit handling.
        """
        ti, tf = self.T
        tn = int(np.ceil((tf - ti) / self.dt))

        Y = np.zeros((tn + 1, 2, X0.shape[1]))
        V = np.zeros((tn + 1, 2, X0.shape[1]))

        Y[0] = X0

        t = ti
        landed = np.zeros(X0.shape[1], dtype=bool)

        for i in range(tn):
            X = Y[i]
            if hasattr(func, "on_land"):
                land_mask = func.on_land(X)
                landed |= land_mask

            k1 = func(X, t)
            k1[:, landed] = 0

            X_temp = X + k1 * self.dt
            k2 = func(X_temp, t + self.dt)
            k2[:, landed] = 0

            Y[i + 1] = X + 0.5 * self.dt * (k1 + k2)
            V[i + 1] = 0.5 * (k1 + k2)
            t += self.dt
        self.V = V
        return Y

    def solve(self, func, method="heun"):
        if method == "heun":
            self.solution = self.heun_method(func, self.x0)
            return self.solution
        else:
            raise ValueError("Unsupported method")

    @property
    def get_solution(self):
        return self.solution

    @property
    def get_velocity(self):
        return self.V

    def plot(
        self,
        alpha=0.1,
        linewidth=0.1,
        s=1,
        save=False,
        filename="trajectory.png",
        *args,
        **kwargs,
    ):
        if not hasattr(self, "solution"):
            raise ValueError("Solution not computed. Call solve first.")
        for i in range(self.solution.shape[2]):
            plt.scatter(
                self.solution[0, 0, i],
                self.solution[0, 1, i],
                c="red",
                label="Initial Position",
                s=s,
            )
            plt.scatter(
                self.solution[-1, 0, i],
                self.solution[-1, 1, i],
                c="blue",
                label="Final Position",
                s=s,
            )
            plt.plot(
                self.solution[:, 0, i],
                self.solution[:, 1, i],
                alpha=alpha,
                linewidth=linewidth,
                *args,
                **kwargs,
            )
        plt.xlabel("X")
        plt.ylabel("Y")
        if save:
            plt.savefig(filename)
        plt.show()

    def plot_map(
        f,
        X,
        V,
        figsize=(15, 12),
        extent=(7, 11, 63.5, 65),
        s=2,
        scatter_color=("blue", "red"),
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

    @staticmethod
    def run_timing_test(velocity_func, T=(0, 10), dt=0.01, Np_values=[10, 100, 1000, 10000]):
        times = []

        for Np in Np_values:
            start_time = time.time()
            traj = Trajectory(T=T, dt=dt, Np=Np)
            traj.initialize_particles(mode="random")
            traj(velocity_func)
            end_time = time.time()
            times.append(end_time - start_time)

        # Calculate linear times for comparison outside the loop
        linear_times = [Np * (times[0] / Np_values[0]) for Np in Np_values]

        Trajectory.plot_times(Np_values, times, linear_times)

    @staticmethod
    def plot_times(Np_values, times, linear_times):
        plt.plot(Np_values, times, marker="o", label="Actual Time")
        plt.plot(
            Np_values, linear_times, marker="o", linestyle="--", label="Linear Time"
        )
        plt.xlabel("Number of Particles (Np)")
        plt.ylabel("Run Time (s)")
        plt.xscale("log")
        plt.yscale("log")
        plt.title("Run Time vs. Number of Particles")
        plt.grid(True)
        plt.legend()
        plt.show()



# ----------------- Velocity func -----------------
def velocity(X, t):
    A, eps, w = 0.10, 0.25, 1  # Initial constants
    a = eps * np.sin(w * t)  # Equation 4a
    b = 1 - 2 * eps * np.sin(w * t)

    x, y = X

    f = a * x**2 + b * x  # Equation 4b
    dx = -np.pi * A * np.sin(np.pi * f) * np.cos(np.pi * y)  # Equation 3
    dy = np.pi * A * np.cos(np.pi * f) * np.sin(np.pi * y) * (2 * a * x + b)

    return np.array([dx, dy])
