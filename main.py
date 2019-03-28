import sys

import sympy as sp

from physics.lagrangian import Lagrangian, Constraint
from physics.pendulum import CompoundPendulumPhysics, n_link_pendulum

if __name__ == "__main__":
    print("")

    physics = CompoundPendulumPhysics(
            L = 1,
            m = 1,
            I = 0
        )

    print("Constructing Pendulum...", end=""); sys.stdout.flush()
    (pendulum, t, x, y, thetas) = n_link_pendulum(2, physics)
    print("Done")

    constraints = [Constraint(x, sp.S.Zero), Constraint(y, sp.S.Zero)]

    print("Calculating Lagrangian...", end=""); sys.stdout.flush() 
    L = pendulum.lagrangian(t)
    print("Done")

    print("Generating Symbolic ODE Equations...", end=""); sys.stdout.flush()
    odeExpressions = L.solve(t, thetas, constraints)
    print("Done")

    print("Converting to Numeric Equations...", end=""); sys.stdout.flush()
    (state_to_y, dy_dt, y_to_state) = odeExpressions.numericize(t)
    print("Done")


    print("")