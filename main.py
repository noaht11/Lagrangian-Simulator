import sys

import sympy as sp
from math import pi

from physics.lagrangian import LagrangianBody, Constraint
from physics.pendulum import CompoundPendulumPhysics, n_link_pendulum
from physics.solver import Solver

sp.init_printing()

def step(text: str):
    print(text, end=""); sys.stdout.flush()

def done():
    print("Done"); sys.stdout.flush()

if __name__ == "__main__":
    print("")

    step("Defining Pendulum...")
    pendulum_physics = CompoundPendulumPhysics(
            L = 5,
            m = 3,
            I = 6
        )
    (pendulum, t, x, y, thetas) = n_link_pendulum(2, pendulum_physics)
    done()

    step("Constructing Lagrangian Body...")
    lagrangian_body = LagrangianBody(t, pendulum, Constraint(y, sp.S.Zero))
    done()

    solver = Solver(lagrangian_body)
    
    step("Converting to Numerical Body...")
    numerical_body = solver.numerical_body()
    done()

    step("Solving for Time Evolution...")
    time_evolver = solver.time_evolver()
    done()

    # step("Calculating Lagrangian...")
    # L = body.lagrangian()
    # done()

    # step("Generating Symbolic ODE Equations...")
    # odeExpressions = L.solve()
    # done()

    # step("Converting to Numerically Solvable Equations...")
    # solver = LagrangianNumericalSolver.from_ode_expr(odeExpressions)
    # done()

    state = [
        0,
        pi/2,
        pi/4,
        0,
        2,
        3
    ]

    # print(solver.state_to_y(0, state))

    print("")