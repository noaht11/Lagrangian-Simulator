from physics.lagrangian import Lagrangian, Constraint
from physics.pendulum import CompoundPendulumPhysics, n_link_pendulum
# from pendulum.core import Coordinates, Physics, MultiPendulum
# from pendulum.impl import CompoundPendulumPhysics
# from pendulum.lagrangian import Lagrangian
# from pendulum.builder import n_link_pendulum

import sympy as sp

if __name__ == "__main__":
    print("")

    physics = CompoundPendulumPhysics(
            L = 1,
            m = 1,
            I = 0
        )

    (pendulum, t, x, y, thetas) = n_link_pendulum(2, physics)

    L = pendulum.lagrangian(t)

    (forces, momenta) = L.forces_and_momenta(t, thetas, [Constraint(x, sp.S.Zero), Constraint(y, sp.S.Zero)])
    
    print("")
    
    # for force in forces:
    #     sp.pprint(force)

    for momentum in momenta:
        sp.pprint(momentum)
        print("")

    print("")