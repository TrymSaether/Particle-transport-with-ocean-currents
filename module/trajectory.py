import numpy as np
from math import pi
import matplotlib.pyplot as plt
import cartopy  # Functions for plotting on map
import cartopy.crs as ccrs
import cartopy.feature as cfeature


class Trajectory:
    def __init__(self):
        self.x0 = 0
        self.dt = 0.01
        self.Np = 100
        self.T = [0, 10]

    def __call__(self, func, x0, T, dt=0.01):
        self.initial_conditions(x0, T, dt)
        return self.solve(func)

    def __repr__(self):
        return f"Trajectory(x0={self.x0}, T={self.T}, dt={self.dt})"

    def get_trajectory(self):
        return self.xs, self.xl, self.ts

    def initial_conditions(self, x0, T, dt=0.01):
        self.x0 = x0
        self.T = T
        self.dt = dt

    def init_random_particles(self, Np=100, loc=[790000, 490000], scale=10000):
        X0 = np.zeros((2, Np))
        X0[0,:] = np.random.normal(loc = loc[0], scale=scale, size = Np)
        X0[1,:] = np.random.normal(loc = loc[1], scale=scale, size = Np)
        self.X0 = X0
        self.initial_conditions(self.X0, self.T, self.dt)
        
    def init_linear_particles(self, Np=100, size=0.1):
        X0 = np.zeros((2, Np))
        X0[0,:] = np.linspace(0, size, Np)
        X0[1,:] = np.linspace(0, size, Np)
        
        self.X0 = X0
        self.initial_conditions(self.X0, self.T, self.dt)
        
    
    def heun(self, func, x0, T, dt=0.01):
        """heun method for solving ODEs

        Args:
            func (X, t): derivative function
            x0: shape (2, Np)
            T: [ti, tf]
            dt: Defaults to 0.01.

        Returns:
            _type_: _description_
        """
        ti, tf = T
        xs = [x0]
        xl = [x0]
        ts = [ti]
        
        while ts[-1] < tf:
            X, t = xs[-1], ts[-1]
            
            if hasattr(func,'on_land'):
                mask = func.on_land(X)
                idx_land = np.where(mask is True)[0]
                xl.append(X[:,idx_land])
                X[:,idx_land] = 0
            
            k1 = func(X, t)
            k2 = func(X + k1 * dt, t + dt)
            xs.append(X + 0.5 * dt * (k1 + k2))
            ts.append(t + dt)
        
        return np.array(xs), np.array(xl), np.array(ts)
    

    def solve(self, func, method="heun"):
        if method == "heun":
            return self.heun(func, self.x0, self.T, self.dt)
        else:
            raise ValueError("Invalid method")

    def integrate(self):  # Alias for solve
        return self.solve(self.func)
    
    
    def plot(self, save=False, *args, **kwargs) -> tuple:
        xs = self.xs
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.plot(xs[0, :], xs[1, :], *args, **kwargs)
        plt.xlabel("X")
        plt.ylabel("Y")

    def plot_map(func, X, color='blue'):
        # scatter plot positions, note the extra transform keyword
        fig = plt.figure(figsize=(9, 6))
        ax = plt.axes(projection=ccrs.NorthPolarStereo())

        ax.add_feature(
            cfeature.NaturalEarthFeature("physical", "land", "10m", color="#cccccc")
        )

        npstere = ccrs.Stereographic(
            central_latitude=90,
            central_longitude=func.dataset.projection_stere.straight_vertical_longitude_from_pole,
            false_easting=func.dataset.projection_stere.false_easting,
            false_northing=func.dataset.projection_stere.false_northing,
            true_scale_latitude=func.dataset.projection_stere.standard_parallel,
        )  # Create projection object for converting particle positions
        ax.scatter(X[0, 0, :], X[0, 1, :], s=1, transform=npstere, label="Initial")
        ax.scatter(
            X[-1, 0, :],
            X[-1, 1, :],
            color=color,
            s=1,
            transform=npstere,
            label="Final",
        )

        # Make outline a bit larger

        # Add gridlines
        gl = ax.gridlines(
            draw_labels=True,
            dms=True,
            x_inline=False,
            y_inline=False,
            alpha=0.5,
            color="k",
            lw=0.5,
        )
        for xs in X:
            plt.plot(
                xs[0, :], xs[1, :], transform=npstere, color="k", lw=0.05, alpha=0.05
            )
        ax.legend()
        plt.tight_layout()
