from pendulum.core import *
from pendulum.lagrangian import *

import sympy as sp

if __name__ == "__main__":
    print("")

    t = sp.symbols("t")
    x = sp.Function("x")
    y = sp.Function("y")
    theta = sp.Function("theta")

    pendulum = CompoundPendulum(
        Pendulum.Coordinates(
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
        ),
        CompoundPendulum.Properties(
            L = 1,
            m = 1,
            I = 0
        )
    )

    L = Lagrangian(pendulum.U(t), pendulum.T(t))

    (forces, momenta) = L.generalized_forces_momenta(t, [theta], [(x, sp.S.Zero), (y, sp.S.Zero)])

    sp.pprint(forces[0])
    print("")
    sp.pprint(momenta[0])

    print("")