from pendulum.core import *

import sympy as sp

if __name__ == "__main__":
    x, y, theta = sp.symbols("x y theta")

    pendulum = Pendulum(
        Pendulum.Properties(
            L = 1,
            m = 1,
            I = 1
        ),
        Pendulum.Symbols(
            x = x,
            y = y,
            theta = theta
        ),
        Pendulum.State(
            x = 0,
            x_dot = 0,
            y = 0,
            y_dot = 0,
            theta = 0,
            theta_dot = 0
        )
    )