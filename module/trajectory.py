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
        Np=10,
        mode="map",
        loc=(0, 0),
        scale=1,
        check_land=False,
    ):
        self.dt = dt
        self.T = T
        self.Np = Np
        self.mode = mode
        self.l0, self.l1 = loc
        self.scale = scale
        self.check_land = check_land
        self.transform = None
        if x0 is not None:
            self.Np = x0.shape[1]
            self.x0 = x0
        else:
            self.x0 = np.zeros((2, Np))
            self.initialize_particles(mode)

    def __call__(self, func, method="heun"):
        self.func = func
        if method == "heun":
            return self.solve(func, method)
        else:
            raise ValueError("Unsupported method")

    def initialize_particles(self, mode):
        if mode == "map":
            self.x0[0, :] = np.random.normal(
                loc=self.l0, scale=self.scale, size=self.Np
            )
            self.x0[1, :] = np.random.normal(
                loc=self.l1, scale=self.scale, size=self.Np
            )

        elif mode == "linear_grid":
            self.x0[0, :] = np.linspace(self.l0, self.l1, self.Np)
            self.x0[1, :] = np.linspace(self.l0, self.l1, self.Np)
        elif mode == "grid":
            n = int(np.sqrt(self.Np))
            x = np.linspace(self.l0, self.l1, n)
            y = np.linspace(self.l0, self.l1, n)
            Xgrid, Ygrid = np.meshgrid(x, y)
            self.Xgrid, self.Ygrid = Xgrid, Ygrid
            self.x0[0, :] = Xgrid.flatten()
            self.x0[1, :] = Ygrid.flatten()

        else:
            raise ValueError("Invalid initialization mode")

    def heun_method(self, func, x0):
        ti, tf = self.T
        tn = int(np.ceil((tf - ti) / self.dt))

        Y = np.zeros((tn, 2, x0.shape[1]))
        V = np.zeros((tn, 2, x0.shape[1]))
        land_percent = np.array([])
        land = np.zeros(x0.shape[1], dtype=bool)
        Y[0] = x0
        t = ti
        for i in range(tn - 1):
            X = Y[i]

            k1 = func(X, t)
            X_temp = X + k1 * self.dt
            k2 = func(X_temp, t + self.dt)

            if self.check_land:
                land_mask = func.on_land(X)
                land |= land_mask

            k1[:, land] = 0
            k2[:, land] = 0
            np.append(land_percent, np.sum(land))

            Y[i + 1] = X + 0.5 * self.dt * (k1 + k2)
            V[i + 1] = 0.5 * (k1 + k2)
            t += self.dt

        self.L = Y[-1, :, land]
        self.land_percent = land_percent / x0.shape[1] * 100
        self.V = V
        return Y

    # Fjern
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

    def get_XY(self):
        X = self.solution[:, 0, :]
        Y = self.solution[:, 1, :]
        return X, Y

    def get_XY_grid(self):
        return self.Xgrid, self.Ygrid

    def get_UV(self):
        U = self.V[:, 0, :]
        V = self.V[:, 1, :]
        return U, V

    def get_land_particles(self):
        LX, LY = self.L.T
        return LX, LY

    def get_land_percent(self):
        return self.land_percent

    def plot(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        X, Y = self.get_XY()
        if self.transform is not None:
            ax.plot(X, Y, transform=self.transform, **kwargs)
        else:
            ax.plot(X, Y, **kwargs)
        ax.set_aspect("equal")
        return ax

    def scatter(self, ax=None, t=0, label="", **kwargs):
        X, Y = self.get_XY()
        if ax is None:
            fig, ax = plt.subplots()
        if self.transform is not None:
            ax.scatter(X[t, :], Y[t, :], transform=self.transform, **kwargs)
            if self.check_land:
                LX, LY = self.get_land_particles()
                ax.scatter(
                    LX[:],
                    LY[:],
                    transform=self.transform,
                    color="red",
                    marker="x",
                    s=25,
                )
        else:
            ax.scatter(X[t, :], Y[t, :], **kwargs)
        return ax

    def streamplot(self, ax=None, **kwargs):
        X, Y = self.get_XY()
        U, V = self.get_UV()
        if ax is None:
            fig, ax = plt.subplots()
        if self.transform is not None:
            ax.streamplot(X, Y, U, V, transform=self.transform, **kwargs)
        else:
            ax.streamplot(X, Y, U, V, **kwargs)
        return ax

    def init_map(self, ax=None, figsize=(10, 10), **kwargs):
        X, Y = self.get_XY()
        U, V = self.get_UV()
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(
            cfeature.NaturalEarthFeature(
                "physical", "land", "10m", edgecolor="face", facecolor="#cccccc"
            )
        )
        ax.coastlines(resolution="10m")
        ax.gridlines(
            draw_labels=True,
            dms=True,
            x_inline=False,
            y_inline=False,
            alpha=0.5,
            color="k",
            lw=0.5,
        )
        projection_params = {
            "central_latitude": 90,
            "central_longitude": self.func.dataset.projection_stere.straight_vertical_longitude_from_pole,
            "false_easting": self.func.dataset.projection_stere.false_easting,
            "false_northing": self.func.dataset.projection_stere.false_northing,
            "true_scale_latitude": self.func.dataset.projection_stere.standard_parallel,
        }
        npstere = ccrs.Stereographic(**projection_params)
        ax.plot(X, Y, transform=npstere, **kwargs)
        ax.set_extent(ax.get_extent(crs=ccrs.PlateCarree()), crs=ccrs.PlateCarree())
        ax.gridlines(
            draw_labels=True,
            dms=True,
            x_inline=False,
            y_inline=False,
            alpha=0.5,
            color="k",
            lw=0.5,
        )
        ax.coastlines(resolution="10m")
        self.transform = npstere
        return ax
