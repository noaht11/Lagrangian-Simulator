from physics.lagrangian import Lagrangian, Constraint
from physics.pendulum import CompoundPendulumPhysics, n_link_pendulum

import sympy as sp

if __name__ == "__main__":
    print("")

    physics = CompoundPendulumPhysics(
            L = 1,
            m = 1,
            I = 0
        )

    (pendulum, t, x, y, thetas) = n_link_pendulum(2, physics)

    constraints = [Constraint(x, sp.S.Zero), Constraint(y, sp.S.Zero)]
    
    L = pendulum.lagrangian(t)
    L.solve(t, thetas, constraints)

    print("")