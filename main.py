import sys

import sympy as sp
from math import pi

from physics.lagrangian import Lagrangian, Constraint
from physics.pendulum import CompoundPendulumPhysics, n_link_pendulum
from physics.solvers import FunctionNumericSolver

def step(text: str):
    print(text, end=""); sys.stdout.flush()
    pass

def done():
    print("Done"); sys.stdout.flush()
    pass

if __name__ == "__main__":
    print("")

    step("Constructing Pendulum...")
    pendulum_physics = CompoundPendulumPhysics(
            L = 5,
            m = 3,
            I = 6
        )
    (pendulum, t, x, y, thetas) = n_link_pendulum(2, pendulum_physics)
    done()

    step("Calculating Lagrangian...")
    constraints = [Constraint(y, sp.S.Zero)]
    L = pendulum.lagrangian(t)
    done()

    step("Generating Symbolic ODE Equations...")
    odeExpressions = L.solve(t, [x] + thetas, constraints)
    done()

    step("Converting to Numerically Solvable Equations...")
    (state_to_y, dy_dt, y_to_state) = odeExpressions.numericize(t)
    solver = FunctionNumericSolver(state_to_y, dy_dt, y_to_state)
    done()

    state = [
        0,
        pi/2,
        pi/4,
        0,
        2,
        3
    ]

    print("")