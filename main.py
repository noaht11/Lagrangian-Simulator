from double_pendulum import *

from math import sin, cos, sqrt

def run(theta1: float = 0, theta2: float = 0, q: float = 0, potential: Potential = ZeroPotential()):
    # Setup pendulum
    L = 1            # m
    m = 2            # kg
    d = sqrt(1/12)*L # m

    pendulum = DoublePendulum(DoublePendulum.Properties(L, m, d), DoublePendulum.State(
        theta1     = theta1,
        theta2     = theta2,
        q          = q,
        theta1_dot = 0,
        theta2_dot = 0,
        q_dot      = 0
    ))

    # Choose behavior
    behavior = GeneralDoublePendulumBehavior(potential)

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
    run(theta1 = pi/100, theta2 = pi/100, q = 0, potential = HarmonicOscillatorPotential(q_k = 1.2358011, q_eq = 1))