from pendulum.core import *
from pendulum.impl import *
from pendulum.lagrangian import *

import sympy as sp

def n_pendulum(n: int, physics: Physics):
    t = sp.symbols("t")
    x = sp.Function("x")
    y = sp.Function("y")

    pendulum = None

    thetas = []
    for i in range(n):
        theta = sp.Function("theta_" + str(i + 1)) # 1-index the thetas
        thetas.append(theta)

        if (pendulum is None):
            coordinates = Coordinates(
                x = x,
                y = y,
                theta = theta
            )
            pendulum = MultiPendulum(coordinates, physics)
        else:
            pendulum.attach_pendulum(t, theta, physics)
    
    return (pendulum, t, x, y, thetas)

if __name__ == "__main__":
    print("")

    physics = CompoundPendulumPhysics(
            L = 1,
            m = 1,
            I = 0
        )

    (pendulum, t, x, y, thetas) = n_pendulum(2, physics)

    L = Lagrangian(pendulum.U(t), pendulum.T(t))

    (forces, momenta) = L.generalized_forces_momenta(t, thetas, [(x, sp.S.Zero), (y, sp.S.Zero)])

    # sp.pprint(forces[0])
    print("")
    sp.pprint(momenta[0])
    sp.pprint(momenta[1])

    print("")