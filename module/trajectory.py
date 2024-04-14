import numpy as np
from matplotlib import pyplot as plt
import time
import cartopy.crs as ccrs
import cartopy.feature as cfeature


class Trajectory:
    """
    Class representing a trajectory of particles.

    Parameters:
    - X0 (ndarray, optional): Initial positions of the particles. If not provided, particles are initialized at (0, 0).
    - time_interval (tuple, optional): Time interval for the trajectory. Default is (0, 10).
    - stepsize (float, optional): Step size for the numerical integration. Default is 0.01.
    - number_of_particles (int, optional): Number of particles in the trajectory. Default is 100.
    - mode (str, optional): Initialization mode for the particles. Default is "map".
    - location (list, optional): Start and end locations for the particles. Default is [0, 0].
    - scale (float, optional): Scale factor for the particle initialization. Default is 1.
    - check_land (bool, optional): Flag indicating whether to check if particles have landed. Default is False.
    """

    def __init__(
        self,
        X0=None,
        time_interval=(0, 10),
        stepsize=0.01,
        number_of_particles=100,
        mode="map",
        location=[0, 0],
        scale=1,
        check_land=False,
    ):
        self.number_of_particles = number_of_particles
        self.time_interval = time_interval
        self.stepsize = stepsize

        self.mode = mode
        self.start_location, self.end_location = location
        self.scale = scale
        self.check_land = check_land
        if X0 is not None:
            self.number_of_particles = X0.shape[1]
            self.X0 = X0
        else:
            self.X0 = np.zeros((2, number_of_particles))
            self.initialize_particles(mode)

    def __call__(self, interpolator, method="heun"):
        """
        Compute the trajectory using the specified interpolator and integration method.

        Parameters:
        - interpolator (object): Interpolator object used to compute the particle velocities.
        - method (str, optional): Integration method. Default is "heun".

        Returns:
        - X (ndarray): X-coordinates of the particle positions.
        - Y (ndarray): Y-coordinates of the particle positions.
        """
        self.interpolator = interpolator
        if method == "heun":
            return self.heun(interpolator, self.X0)
        else:
            raise ValueError("Unsupported method")

    def initialize_particles(self, mode):
        """
        Initialize the particles based on the specified mode.

        Parameters:
        - mode (str): Initialization mode for the particles.
        """
        if mode == "map":
            self.X0[0, :] = np.random.normal(
                loc=self.start_location, scale=self.scale, size=self.number_of_particles
            )
            self.X0[1, :] = np.random.normal(
                loc=self.end_location, scale=self.scale, size=self.number_of_particles
            )

        elif mode == "linear_grid":
            self.X0[0, :] = np.linspace(
                self.start_location, self.end_location, self.number_of_particles
            )
            self.X0[1, :] = np.linspace(
                self.start_location, self.end_location, self.number_of_particles
            )
        elif mode == "grid":
            n = int(np.sqrt(self.number_of_particles))
            if n**2 != self.number_of_particles:
                raise ValueError("Invalid number of particles")
            x = np.linspace(self.start_location, self.end_location, n)
            y = np.linspace(self.start_location, self.end_location, n)
            grid_X, grid_Y = np.meshgrid(x, y)
            self.X0[0, :] = grid_X.flatten()
            self.X0[1, :] = grid_Y.flatten()

        else:
            raise ValueError("Invalid initialization mode")

    def heun(self, interpolator, X0):
        """
        Perform numerical integration using the Heun's method.

        Parameters:
        - interpolator (object): Interpolator object used to compute the particle velocities.
        - X0 (ndarray): Initial positions of the particles.

        Returns:
        - X (ndarray): X-coordinates of the particle positions.
        - Y (ndarray): Y-coordinates of the particle positions.
        """
        ti, tf = self.time_interval
        h = self.stepsize
        tn = int(np.ceil((tf - ti) / h))
        landed = np.zeros(X0.shape[1], dtype=bool)
        percent_landed = []

        VELOCITY = np.zeros((tn, 2, X0.shape[1]))
        SOLUTION = np.zeros((tn, 2, X0.shape[1]))
        SOLUTION[0] = X0
        t = ti
        for i in range(tn - 1):
            X = SOLUTION[i]

            K1 = interpolator(X, t)
            K1 = interpolator(X + K1 * h, t + h)

            if self.check_land:
                landed |= interpolator.on_land(X)
                percent_landed.append(np.sum(landed))

            K1[:, landed] = 0
            K1[:, landed] = 0

            SOLUTION[i + 1] = X + 0.5 * h * (K1 + K1)
            VELOCITY[i + 1] = 0.5 * (K1 + K1)
            t += h

        if self.check_land:
            self.land_percent = np.array(percent_landed) / X0.shape[1] * 100
            self.X_land, self.Y_land = SOLUTION[:, 0, landed], SOLUTION[:, 1, landed]

        self.X, self.Y = SOLUTION[:, 0, :], SOLUTION[:, 1, :]
        self.U, self.V = VELOCITY[:, 0, :], VELOCITY[:, 1, :]
        return self.X, self.Y

    def get_XY(self):
        """
        Returns the X and Y coordinates for the trajectory.

        If the interpolator does not have the attribute "on_land", it generates a grid of X and Y coordinates
        based on the limits of the trajectory. Otherwise, it returns the pre-calculated X and Y coordinates.

        Returns:
            gridX (ndarray): The X coordinates of the trajectory.
            gridY (ndarray): The Y coordinates of the trajectory.
        """
        if not hasattr(self.interpolator, "on_land"):
            limits = self.get_limits()
            xmin, xmax, ymin, ymax = limits
            gridX, gridY = np.meshgrid(
                np.linspace(xmin, xmax, 100), np.linspace(ymin, ymax, 100)
            )
            return gridX, gridY
        else:
            return self.X, self.Y
    """
    def get_XY_land(self):
        
        Returns the X and Y coordinates of the landing point.

        If the check_land attribute is True, the method returns the X_land and Y_land attributes.
        If the check_land attribute is False, the method returns None.

        Returns:
            tuple or None: A tuple containing the X and Y coordinates of the landing point, or None if check_land is False.
        
        if self.check_land:
            return self.X_land, self.Y_land
        return None
    """
    def get_UV(self, t0=0):
        """
        Returns the U and V components of the wind velocity at a given time.

        Parameters:
            t0 (float): The time at which to retrieve the wind velocity. Default is 0.

        Returns:
            tuple: A tuple containing the U and V components of the wind velocity.
                   If the interpolator has the attribute "on_land", it returns the interpolated wind velocity at the given time.
                   Otherwise, it returns the pre-calculated U and V components.

        """
        if not hasattr(self.interpolator, "on_land"):
            gridX, gridY = self.get_XY()
            return self.interpolator([gridX, gridY], t0)
        return self.U, self.V

    def get_land_percent(self):
        """
        Returns the percentage of particles that have landed.

        Returns:
            ndarray: An array containing the percentage of particles that have landed at each time step.
        """
        return self.land_percent

    def get_time(self):
        """
        Returns the time array for the trajectory.

        Returns:
            ndarray: An array containing the time values for the trajectory.
        """
        return np.arange(self.time_interval[0], self.time_interval[1], self.stepsize)

    def get_limits(self):
        """
        Returns the minimum and maximum values of X and Y coordinates.

        Returns:
            tuple: A tuple containing the minimum and maximum values of X and Y coordinates.
        """
        return np.min(self.X), np.max(self.X), np.min(self.Y), np.max(self.Y)

    def distance_particles(self):
        """
        Calculate the distance traveled by each particle.

        Returns:
            ndarray: An array containing the distance traveled by each particle.
        """
        return np.sum(
            np.sqrt(np.diff(self.X, axis=0) ** 2 + np.diff(self.Y, axis=0) ** 2), axis=0
        )

    def info(self):
        """
        Print information about the trajectory.
        """
        print(f"Trajectory with:")
        print(f"Np: {self.number_of_particles} particles")
        print(f"Time: {self.time_interval}")
        print(f"Time step: {self.stepsize}")
        print(f"Mode: {self.mode}")
        print(f"Check land: {self.check_land}")
        print(f"Velocity shape: {self.V.shape}")
        print(f"X, Y shape: {self.get_XY()[0].shape}")
        print(f"U, V shape: {self.get_UV()[0].shape}")
        print(f"xmin: {np.min(self.X)}, xmax: {np.max(self.X)}")
        print(f"ymin: {np.min(self.Y)}, ymax: {np.max(self.Y)}")
        print(f"Distance mean: {np.mean(self.distance_particles())}")
        if hasattr(self, "land_percent"):
            print(f"Land percent: {self.land_percent[-1]}%")
        if hasattr(self, "npstere"):
            print(f"Projection: {self.npstere}")
        if hasattr(self, "X_land"):
            print(f"Land particles: {self.X_land.shape[0]}")
        if hasattr(self.interpolator, "on_land"):
            print(f"Windage factor: {self.interpolator.fw}")
        return None

    def plot(self, axs=None, **kwargs):
        """
        Plot the trajectory.

        Parameters:
            axs (matplotlib.axes.Axes, optional): The axes on which to plot the trajectory. If not provided, a new figure and axes will be created.
            **kwargs: Additional keyword arguments to be passed to the plot function.

        Returns:
            matplotlib.axes.Axes: The axes on which the trajectory is plotted.
        """
        if axs is None:
            figs, axs = plt.subplots()

        if hasattr(self, "npstere"):
            axs.plot(self.X, self.Y, transform=self.npstere, **kwargs)
        else:
            axs.plot(self.X, self.Y, **kwargs)
        axs.set_aspect("equal")
        return axs

    def scatter(self, axs=None, t=0, label="", **kwargs):
        """
        Scatter plot of the trajectory at a given time.

        Parameters:
            axs (matplotlib.axes.Axes, optional): The axes on which to plot the scatter plot. If not provided, a new figure and axes will be created.
            t (int, optional): The time index at which to plot the scatter plot. Default is 0.
            label (str, optional): The label for the scatter plot. Default is an empty string.
            **kwargs: Additional keyword arguments to be passed to the scatter function.

        Returns:
            matplotlib.axes.Axes: The axes on which the scatter plot is plotted.
        """
        if axs is None:
            figs, axs = plt.subplots()

        if hasattr(self, "npstere"):
            axs.scatter(self.X[t, :], self.Y[t, :], transform=self.npstere, **kwargs)

        if self.check_land:
            axs.scatter(
                self.X_land[-1, :],
                self.Y_land[-1, :],
                transform=self.npstere,
                label=label,
                color="red",
                marker="x",
                s=15,
            )
            axs.scatter(self.X[t, :], self.Y[t, :], **kwargs)
        else:
            axs.scatter(self.X[t, :], self.Y[t, :], **kwargs)
        axs.set_aspect("equal")
        return axs

    def streamplot(self, axs=None,time_velocity_field=0, **kwargs):
        """
        Plots streamlines of the vector field defined by the trajectory.

        Parameters:
            axs (matplotlib.axes.Axes, optional): The axes on which to plot the streamlines. If not provided, a new figure and axes will be created.
            **kwargs: Additional keyword arguments to be passed to the `streamplot` function.

        Returns:
            matplotlib.axes.Axes: The axes object on which the streamlines are plotted.
        """
        if axs is None:
            figs, axs = plt.subplots()
        if hasattr(self, "npstere"):
            axs.streamplot(
                self.X, self.Y, self.U, self.V, transform=self.npstere, **kwargs
            )
        if not hasattr(self.interpolator, "on_land"):
            gridX, gridY = self.get_XY()
            gridU, gridV = self.get_UV(time_velocity_field)
            axs.streamplot(gridX, gridY, gridU, gridV, **kwargs)
        else:
            axs.streamplot(self.X, self.Y, self.U, self.V, **kwargs)
        return axs

    def init_map(self, axs=None, figsize=(10, 10), **kwargs):
        """
        Initialize the map for plotting trajectory.

        Parameters:
        - axs (matplotlib.axes.Axes, optional): The axes object to plot on. If None, a new figure and axes will be created.
        - figsize (tuple, optional): The size of the figure in inches. Default is (10, 10).
        - **kwargs: Additional keyword arguments to be passed to the plot function.

        Returns:
        - axs (matplotlib.axes.Axes): The axes object used for plotting.

        """
        if axs is None:
            figs, axs = plt.subplots(
                figsize=figsize, subplot_kw={"projection": ccrs.PlateCarree()}
            )

        axs.add_feature(
            cfeature.NaturalEarthFeature(
                "physical", "land", "10m", edgecolor="face", facecolor="#cccccc"
            )
        )
        projection_params = {
            "central_latitude": 90,
            "central_longitude": self.interpolator.dataset.projection_stere.straight_vertical_longitude_from_pole,
            "false_easting": self.interpolator.dataset.projection_stere.false_easting,
            "false_northing": self.interpolator.dataset.projection_stere.false_northing,
            "true_scale_latitude": self.interpolator.dataset.projection_stere.standard_parallel,
        }
        npstere = ccrs.Stereographic(**projection_params)
        axs.plot(self.X, self.Y, transform=npstere, **kwargs)
        axs.set_extent(axs.get_extent(crs=ccrs.PlateCarree()), crs=ccrs.PlateCarree())
        axs.gridlines(
            draw_labels=True,
            dms=True,
            x_inline=False,
            y_inline=False,
            alpha=0.5,
            color="k",
            lw=0.5,
        )
        axs.coastlines(resolution="10m")
        self.npstere = npstere
        return axs
