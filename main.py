from double_pendulum import *

from math import sin, cos, sqrt

def setup_pendulum(theta1: float = 0, theta2: float = 0, q: float = 0, theta1_dot: float = 0, theta2_dot: float = 0, q_dot: float = 0, L: float = 1, m: float = 1):
    d = sqrt(1/12)*L # m
    pendulum = DoublePendulum(DoublePendulum.Properties(L, m, d), DoublePendulum.State(
        theta1     = theta1,
        theta2     = theta2,
        q          = q,
        theta1_dot = theta1_dot,
        theta2_dot = theta2_dot,
        q_dot      = q_dot
    ))
    return pendulum

def run_potential(theta1: float = 0, theta2: float = 0, q: float = 0, potential: Potential = ZeroPotential()):
    run(
        setup_pendulum(theta1 = theta1, theta2 = theta2, q = q),
        GeneralDoublePendulumBehavior(potential)
    )

def run_forcing(theta1: float = 0, theta2: float = 0, forcing_function: QForcingFunction = FixedQForcingFunction(), potential: Potential = ZeroPotential()):
    run(
        setup_pendulum(theta1 = theta1, theta2 = theta2, q = forcing_function.q(0), q_dot = forcing_function.dq_dt(0)),
        ForcedQDoublePendulumBehavior(forcing_function, potential)
    )

def run(pendulum: DoublePendulum, behavior: DoublePendulumBehavior):
    # Setup solvers
    time_evolver = ODEINTTimeEvolver()
    simulation = DoublePendulumSimulation(pendulum, behavior, time_evolver)

    # Simulation parameters
    dt      = 1.0 / 50
    draw_dt = 1.0 / 50

    # Run animated simulation
    animator = DoublePendulumAnimator(simulation)
    animator.init()
    animator.run(dt, draw_dt, 1000)

if __name__ == "__main__":
    # run_potential(theta1 = pi/100, theta2 = pi/100, q = 0, potential = FixedQPotential())
    run_forcing(theta1 = pi/100, theta2 = pi/100, forcing_function = SinusoidalForcing(amplitude = 0.05, frequency = 10))