import sys
from math import pi

import sympy as sp
import numpy as np

from physics.lagrangian import LagrangianBody, DegreeOfFreedom, Constraint
import physics.pendulum
from physics.pendulum import CompoundPendulumPhysics
from physics.pendulum import SinglePendulumLagrangianPhysics, SinglePendulum, SinglePendulumSolver
from physics.pendulum import MultiPendulum, MultiPendulumSolver
from physics.animation import PhysicsAnimation

sp.init_printing()

def step(text: str):
    print(text, end=""); sys.stdout.flush()

def done():
    print("Done"); sys.stdout.flush()

def create_init_state(n: int, theta: float) -> np.ndarray:
    qs = np.ones(n, dtype = np.float32) * theta
    q_dots = np.zeros(n, dtype = np.float32)

    return np.concatenate((qs, q_dots))

def create_multi_pendulum_params(n: int, L: float, m: float, I: float, k: float) -> np.ndarray:
    params = np.array([], dtype=np.float32)
    for i in range(n):
        params = np.append(params, np.array([L, m, I, k]))
    return params

def create_forcing(t: sp.Symbol, A: sp.Symbol, f: sp.Symbol) -> sp.Expr:
    return A * sp.cos(2*pi*f * t)

def update_k(current_params: np.ndarray, k: float, n) -> np.ndarray:
    for i in range(n):
        current_params[3 + 4*i] = k
    return current_params

def update_A(current_params: np.ndarray, A: float) -> np.ndarray:
    current_params[-2] = A
    return current_params

def update_f(current_params: np.ndarray, f: float) -> np.ndarray:
    current_params[-1] = f
    return current_params

###################################################################################################################################################################################
# MULTI PENDULUM
###################################################################################################################################################################################

if __name__ == "__main__":
    print("")

    step("Defining Pendulum...")

    ########################### CONFIG ###########################
    n = 2

    L = 0.045
    m = 0.003
    I = 1/12*m*L**2
    k = 0#2E-5

    theta = pi/10

    A = 0.0
    f = 0.0
    ########################### CONFIG ###########################

    params = create_multi_pendulum_params(n, L, m, I, k)
    params = np.concatenate((params, np.array([A, f])))

    (pendulum_lagrangian_physics, t, x, y) = physics.pendulum.n_link_pendulum(n, physics.pendulum.compound_pendulum_physics_generator)
    done()

    step("Constructing Lagrangian Body...")
    A_sym, f_sym = sp.symbols("A f")
    pendulum = MultiPendulum(t, pendulum_lagrangian_physics, Constraint(x, sp.S.Zero), Constraint(y, create_forcing(t, A_sym, f_sym), [A_sym, f_sym]))
    done()

    solver = MultiPendulumSolver(pendulum)
    
    step("Generating Simulation...")
    init_state = create_init_state(n, theta)
    simulation = solver.simulate(init_state, params)
    done()

    step("Generating Artist...")
    artist = solver.artist()
    done()

    step("Generating Animation...")
    animation = PhysicsAnimation(simulation, artist, PhysicsAnimation.AnimationConfig(
        size = L*4,
        en_reset = True,
        en_start_stop = True
    ), [
        PhysicsAnimation.Parameter("k", 0, 1, 0.01, k, lambda current_params, new_k: update_k(current_params, new_k*10E-5, n)),
        PhysicsAnimation.Parameter("f (Hz)", 0, 500, 1, f, lambda current_params, new_f: update_f(current_params, new_f)),
        PhysicsAnimation.Parameter("A (cm)", 0, 2, 0.1, A, lambda current_params, new_A: update_A(current_params, new_A*1E-2))
    ])
    done()

    animation.init()
    dt = 1/400
    draw_dt = dt
    animation.run(dt, draw_dt, -1)

    print("")