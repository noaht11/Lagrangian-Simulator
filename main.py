import sys
from math import pi

import argparse
import sympy as sp
import numpy as np

from physics.lagrangian import LagrangianBody, DegreeOfFreedom, Constraint
from physics.simulation import PhysicsSimulation
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

def update_k(simulation: PhysicsSimulation, k: float, n: int):
    current_params = simulation.parameters
    for i in range(n):
        current_params[3 + 4*i] = k
    simulation.parameters = current_params

def update_A(simulation: PhysicsSimulation, A: float):
    current_params = simulation.parameters
    current_params[-2] = A
    simulation.parameters = current_params

def update_f(simulation: PhysicsSimulation, f: float):
    current_params = simulation.parameters
    current_params[-1] = f
    simulation.parameters = current_params

def update_init_state(simulation: PhysicsSimulation, theta: float, n: int):
    simulation.init_state = create_init_state(n, theta)

###################################################################################################################################################################################
# MAIN
###################################################################################################################################################################################

if __name__ == "__main__":
    print("")

    #### CONSTANTS
    FRICTION_SCALE_FACTOR = 1E-4
    AMPLITUDE_SCALE_FACTOR = 1E-2

    ### DEFAULT CONFIG
    n = 1

    L = 0.045
    m = 0.003
    I = 1/12*m*L**2
    k = 0 * FRICTION_SCALE_FACTOR

    theta = pi/10

    A = 0.0 * AMPLITUDE_SCALE_FACTOR
    f = 0.0

    dt = 1/400
    duration = -1

    interactive = False

    output = None
    output_fps = int(1/dt)

    #### CMD LINE ARGS
    parser = argparse.ArgumentParser(description="Pendulum Simulation")
    # Required
    parser.add_argument("n", metavar="n_links", type=int, default=n, help="Number of links in the pendulum")
    # Optional General
    parser.add_argument("-i", "--interactive", action="store_true", help="Set this flag to enable interactive mode")
    parser.add_argument("-dt", type=float, default=dt, help="Time step for the simulation in seconds")
    parser.add_argument("-d", "--duration", type=float, default=duration, help="Duration for which to run the simulation in seconds. Pass -1 to run indefinitely. If interactive is set, this will be ignored.")
    parser.add_argument("-o", "--output", type=str, default=output, help="File path for saving a gif of the animation")
    parser.add_argument("-fps", type=int, default=output_fps, help="FPS for the gif")
    # Optional Pendulum Config
    parser.add_argument("-k", "--friction", type=float, default=(k / FRICTION_SCALE_FACTOR), help="Friction coefficient (between 0 and 1)")
    parser.add_argument("-theta", type=float, default=theta, help="Initial angle in radians")
    parser.add_argument("-A", "--amplitude", type=float, default=(A / AMPLITUDE_SCALE_FACTOR), help="Amplitude of vertical oscillation in cm")
    parser.add_argument("-f", "--frequency", type=float, default=f, help="Frequency of vertical oscillation in Hz")

    args = parser.parse_args()
    n = args.n
    k = args.friction * FRICTION_SCALE_FACTOR
    theta = args.theta
    A = args.amplitude * AMPLITUDE_SCALE_FACTOR
    f = args.frequency
    dt = args.dt
    duration = args.duration
    interactive = args.interactive
    output = args.output
    output_fps = args.fps

    ########################### MAIN ###########################
    step("Defining Pendulum...")

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
        size = L*(n+1),
        mode = (PhysicsAnimation.AnimationConfig.MODE_INTERACTIVE if interactive else PhysicsAnimation.AnimationConfig.MODE_AUTONOMOUS),
        en_reset = True,
        en_start_stop = True,
        save_gif_path = output,
        save_fps = output_fps
    ), [
        PhysicsAnimation.Parameter("k", 0, 1, 0.01, k, lambda new_k: update_k(simulation, new_k*FRICTION_SCALE_FACTOR, n)),
        PhysicsAnimation.Parameter("f (Hz)", 0, 500, 1, f, lambda new_f: update_f(simulation, new_f)),
        PhysicsAnimation.Parameter("A (cm)", 0, 2, 0.01, A, lambda new_A: update_A(simulation, new_A*AMPLITUDE_SCALE_FACTOR)),
        PhysicsAnimation.Parameter("theta (rad)", -pi, pi, 0.01, theta, lambda new_theta: update_init_state(simulation, new_theta, n))
    ])
    done()

    animation.init()
    draw_dt = dt
    animation.run(dt, draw_dt, duration)

    print("")