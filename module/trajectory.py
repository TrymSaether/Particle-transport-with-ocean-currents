import numpy as np
import matplotlib.pyplot as plt


class Trajectory:
    def __init__(self):
        self.x0 = 0
        self.dt = 0.01
        self.Np = 100
        self.t = [0, 10]

    def __call__(self, x0, t, dt=0.01):
        self.initial_conditions(x0, t, dt)
        return self.integrate()

    def __repr__(self):
        return f"Trajectory(x0={self.x0}, t={self.t}, dt={self.dt})"

    def get_trajectory(self):
        return self.xs

    def initial_conditions(self, x0, t, dt=0.01):
        self.x0 = x0
        self.t = t
        self.dt = dt

    def init_particles(self, Np=100):
        self.x0 = np.zeros_like((2, Np))
        self.x0[0, :] = np.random.normal(loc=790000, scale=10000, size=Np)
        self.x0[1, :] = np.random.normal(loc=490000, scale=10000, size=Np)

    def heun(self, func, x0, t, dt=0.01) -> tuple:
        ti, tf = t
        xs = [x0]
        ts = [ti]

        while ts[-1] < tf:
            X, t = xs[-1], ts[-1]
            k1 = func(X, t)
            k2 = func(X + k1 * dt, t + dt)

            xs.append(X + 0.5 * dt * (k1 + k2))
            ts.append(t + dt)

        self.xs, self.ts = np.array(xs), np.array(ts)
        return self.xs, self.ts

    def solve(self, func, method="heun") -> tuple:
        if method == "heun":
            return self.heun(func, self.x0, self.t, self.dt)
        if func is None:
            raise ValueError("Function must be defined")
        else:
            raise ValueError("Invalid method")

    def plot(self, save=False, *args, **kwargs) -> tuple:
        xs = self.xs
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.plot(xs[0, :], xs[1, :], *args, **kwargs)
        plt.xlabel("X")
        plt.ylabel("Y")
        

    def integrate(self):
        # Alias for solve
        return self.solve(self.func)
