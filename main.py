from pendulum.core import Coordinates, Physics, MultiPendulum
from pendulum.impl import CompoundPendulumPhysics
from pendulum.lagrangian import Lagrangian
from pendulum.builder import n_link_pendulum

import sympy as sp

if __name__ == "__main__":
    print("")

    physics = CompoundPendulumPhysics(
            L = 1,
            m = 1,
            I = 0
        )

    (pendulum, t, x, y, thetas) = n_link_pendulum(3, physics)

    L = Lagrangian(pendulum.U(t), pendulum.T(t))

    (forces, momenta) = L.generalized_forces_momenta(t, thetas, [(x, sp.S.Zero), (y, sp.S.Zero)])
    
    print("")
    
    for momentum in momenta:
        sp.pprint(momentum)

    print("")