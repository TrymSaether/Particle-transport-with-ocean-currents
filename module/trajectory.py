import numpy as np
from matplotlib import pyplot as plt
import time
import cartopy.crs as ccrs
import cartopy.feature as cfeature


class Trajectory:
    def __init__(
        self,
        x0=None,
        T=(0, 10),
        dt=0.01,
        Np=100,
        size=0.1,
        loc=(790000, 490000),
        scale=10000,
        mode="random",
    ):
        self.dt = dt
        self.T = T
        self.Np = Np
        self.loc = loc
        self.scale = scale
        self.size = size

        if x0 is not None:
            self.x0 = x0
        else:
            self.x0 = np.zeros((2, Np))
            self.initialize_particles(mode)

    def __call__(self, func, method="heun"):
        return self.solve(func, method)

    def initialize_particles(self, mode="random"):
        if mode == "random":
            self.x0[0, :] = np.random.normal(
                loc=self.loc[0], scale=self.scale, size=self.Np
            )
            self.x0[1, :] = np.random.normal(
                loc=self.loc[1], scale=self.scale, size=self.Np
            )
        elif mode == "linear":
            self.x0[0, :] = np.linspace(1e-3, self.size, self.Np)
            self.x0[1, :] = np.linspace(1e-3, self.size, self.Np)
        elif mode == "grid":
            self.x0 = np.meshgrid(
                np.linspace(1e-3, self.size, int(np.sqrt(self.Np))),
                np.linspace(1e-3, self.size, int(np.sqrt(self.Np))),
            )
            self.x0 = np.array(self.x0).reshape(2, self.Np)

        else:
            raise ValueError("Invalid initialization mode")

    def heun_method(self, func, X0, check_land=False):
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
        land_percent = np.array([])
        # [0,0,1,1,0,1,0]
        for i in range(tn):
            X = Y[i]
            if hasattr(func, "on_land") and check_land:
                land_mask = func.on_land(X)
                landed |= land_mask

            percent = np.sum(landed)
            np.append(land_percent, percent)

            k1 = func(X, t)
            k1[:, landed] = 0
            X_temp = X + k1 * self.dt
            k2 = func(X_temp, t + self.dt)
            k2[:, landed] = 0

            Y[i + 1] = X + 0.5 * self.dt * (k1 + k2)
            V[i + 1] = 0.5 * (k1 + k2)

            t += self.dt

        self.land_percent = land_percent / self.Np
        self.V = V
        self.L = Y[-1, :, landed]

        return Y

    def solve(self, func, method="heun"):
        if method == "heun":
            self.solution = self.heun_method(func, self.x0)
            return self.solution
        else:
            raise ValueError("Unsupported method")

    def get_solution(self):
        return self.solution

    def get_velocity(self):
        return self.V

    def get_land_particles(self):
        return self.L.T

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
            s = 3 * s
            ax.scatter(
                L[0, :],
                L[1, :],
                transform=npstere,
                s=s,
                label="Landed particles",
                color=land_color,
            )
        # Setting the map extent and adding gridlines

        ax.legend()
        plt.tight_layout()

        plt.show()
