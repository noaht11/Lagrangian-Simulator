import sys
from math import pi

import sympy as sp
import numpy as np

from physics.lagrangian import LagrangianBody, DegreeOfFreedom, Constraint
from physics.pendulum import CompoundPendulumPhysics
from physics.pendulum import SinglePendulumLagrangianPhysics, SinglePendulum, SinglePendulumSolver
from physics.pendulum import n_link_pendulum, MultiPendulum, MultiPendulumSolver
from physics.animation import PhysicsAnimation

sp.init_printing()

def step(text: str):
    print(text, end=""); sys.stdout.flush()

def done():
    print("Done"); sys.stdout.flush()

def create_init_state(n: int, theta: float):
    qs = np.ones(n, dtype = np.float32) * theta
    q_dots = np.zeros(n, dtype = np.float32)

    return np.concatenate((qs, q_dots))

def create_forcing(A: float, f: float):
    return A * sp.cos(2*pi*f * t)

###################################################################################################################################################################################
# MULTI PENDULUM
###################################################################################################################################################################################

if __name__ == "__main__":
    print("")

    n = 2
    theta = pi/10

    step("Defining Pendulum...")
    L = 0.045
    m = 0.003
    I = 1/12*m*L**2
    single_pendulum_physics = CompoundPendulumPhysics(
            L = L,
            m = m,
            I = I,
            k = 2E-5
        )
    (pendulum_lagrangian_physics, t, x, y, thetas) = n_link_pendulum(n, single_pendulum_physics)
    done()

    step("Constructing Lagrangian Body...")
    pendulum = MultiPendulum(t, pendulum_lagrangian_physics, Constraint(x, create_forcing(0.0, 140/(2*pi))), Constraint(y, create_forcing(0.0, 140/(2*pi))))
    done()

    solver = MultiPendulumSolver(pendulum)
    
    step("Generating Simulation...")
    init_state = create_init_state(n, theta)
    simulation = solver.simulate(init_state)
    done()

    step("Generating Artist...")
    artist = solver.artist()
    done()

    step("Generating Animation...")
    animation = PhysicsAnimation(simulation, artist)
    done()

    animation.init()
    dt = 1/400
    draw_dt = dt
    animation.run(dt, draw_dt, 10000)

    print("")