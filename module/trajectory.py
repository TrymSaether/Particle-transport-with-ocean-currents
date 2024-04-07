import numpy as np
from math import pi

import matplotlib.pyplot as plt


class Trajectory:
    def __init__(self):
        self.A, self.EPS, self.W = 0.1, 0.25, 1
        self.x0, self.y0 = 0, 0
        self.t0, self.tf = 0, 0
        self.dt = 0.01
        self.Np = 10

    def __call__(self, x0, y0, T, dt=0.01):
        self.initial_conditions(x0, y0, T, dt)
        return self.integrate()

    def initial_conditions(self, x0, y0, T,Np, dt=0.01):
        self.t0, self.tf = T # [0, 6*24*60*60]
        self.xs, self.ys, self.ts = [x0], [y0], [self.t0]
        self.dt = dt
        self.Np = Np

    def integrate(self, method="heuns"):
        if method == "heuns":
            return self.heuns()
        else:
            raise ValueError("Invalid method")
    
    def heuns(self, func, X, T):
        t0, tf, dt = self.t0, self.tf, self.dt
        h = dt
        XS = np.array(X)
        
        for t in T:
            X = XS[-1]
            k1 = func(X, t)
            k2 = func(X + k1*h, t + h)
            np.append(XS, X + 0.5 * h * (k1 + k2))
            
        return np.array(self.XS), np.array(self.ts)

    

    def plot_position(self, xs, ys, traj_color="blue", traj_linestyle="-", label=None, save=False, show=True, title=None):
        plt.plot(xs, ys, label=label, color=traj_color, linestyle=traj_linestyle)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(title)
        if save:
            plt.savefig("trajectory.png")
        if show:
            plt.show()

    def oppgave_1a(self, Ni=100, Nf=10000, steps=5):
        def func(self, x, t):
            a = self.EPS * np.sin(self.W * t)
            b = 1 - 2 * self.EPS * np.sin(self.W * t)
            f = a * x**2 + b * x
            df_dx = 2 * a * x + b
            return f, df_dx

        def velocity_analytisk(self, x, y, t):
            f, df_dx = self.func(x, t)
            vx = -pi * self.A * np.sin(pi * f) * np.cos(pi * y)
            vy = pi * self.A * np.cos(pi * f) * np.sin(pi * y) * (df_dx)
            return vx, vy
        T = np.array([0, 50])
        x0, y0 = np.array([1.05, 0.50])
        ti, tf = T
        xs, ys, ts = self(x0, y0, T, Nf)
        N = np.linspace(Ni, Nf, steps)
        for n in N:
            xs, ys, ts = self(x0, y0, T, int(n))
            dt = (tf - ti) / n
            plt.plot(xs, ys, label="dt = {}".format(round(dt,4)))
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Oppgave 1a: Posisjonen til partikkelen for forskjellige dt (nøyaktighet)")
        plt.legend()
        plt.grid()
        plt.show()
        
    def oppgave_1b(self, Np=100):
        T = np.array([0, 10])  # Change the time interval to [0, 10]
        x0, y0 = np.meshgrid(np.linspace(0, 0.1, int(np.sqrt(Np))), np.linspace(0, 0.1, int(np.sqrt(Np))))
        x0, y0 = x0.flatten(), y0.flatten()
        final_positions = []
        for i in range(len(x0)):
            xs, ys, ts = self(x0[i], y0[i], T, Np)
            plt.plot(xs, ys, color='gray', alpha=0.5)  # Plot the trajectory
            final_positions.append((xs[-1], ys[-1]))
        final_positions = np.array(final_positions)
        plt.scatter(x0, y0, color='blue', label='Initial positions')  # Scatter plot of initial positions
        plt.scatter(final_positions[:, 0], final_positions[:, 1], color='red', label='Final positions')  # Scatter plot of final positions
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Oppgave 1b: Posisjonen til partikkelen for forskjellige startverdier")
        plt.legend()
        plt.grid()
        plt.show()
    
