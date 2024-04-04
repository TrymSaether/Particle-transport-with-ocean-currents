import numpy as np
from math import pi

import matplotlib.pyplot as plt

class Trajectory:
    def __init__(self):
        self.A, self.EPS, self.W = 0.1, 0.25, 1
        self.x0, self.y0 = 0, 0
        self.t0, self.tf = 0, 0
        self.Nmax = 100

    def __call__(self, x0, y0, T, Nmax=100):
        self.initial_conditions(x0, y0, T, Nmax)
        return self.integrate()

    def initial_conditions(self, x0, y0, T, Nmax=100):
        self.t0, self.tf = T
        self.xs, self.ys, self.ts = [x0], [y0], [self.t0]
        self.Nmax = Nmax
        self.dt = (self.tf - self.t0) / self.Nmax

    def integrate(self, method="heuns"):
        if method == "heuns":
            return self.heuns()
        else:
            raise ValueError("Invalid method")

    def heuns(self):
        _, tf, dt = self.t0, self.tf, self.dt
        while self.ts[-1] < tf:
            x,y,t = self.xs[-1],self.ys[-1],self.ts[-1]
            k0_x, k0_y = self.velocity(x, y, t)
            k1_x, k1_y  = self.velocity(x + dt * k0_x, y + dt*k0_y, t + dt)
            self.xs.append(x + 0.5 * dt * (k0_x + k1_x))
            self.ys.append(y + 0.5 * dt * (k0_y + k1_y))
            self.ts.append(t + dt)
        return np.array(self.xs), np.array(self.ys), np.array(self.ts)

    def func(self, x, t):
        a = self.EPS * np.sin(self.W * t)
        b = 1 - 2 * self.EPS * np.sin(self.W * t)
        f = a * x**2 + b * x
        df_dx = 2 * a * x + b
        return f, df_dx

    def velocity(self, x, y, t):
        f, df_dx = self.func(x, t)
        vx = -pi * self.A * np.sin(pi * f) * np.cos(pi * y)
        vy = pi * self.A * np.cos(pi * f) * np.sin(pi * y) * (df_dx)
        return vx, vy

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
    
    
