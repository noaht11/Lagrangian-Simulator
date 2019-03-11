from double_pendulum import *

from math import sin, cos, sqrt

def run(theta1: float, theta2: float, behavior: DoublePendulumBehavior):
    # Setup pendulum
    L = 1            # m
    m = 2            # kg
    d = sqrt(1/12)*L # m

    pendulum = DoublePendulum(DoublePendulum.Properties(L, m, d), DoublePendulum.State(
        theta1     = theta1,
        theta2     = theta2,
        q          = 0,
        theta1_dot = 0,
        theta2_dot = 0,
        q_dot      = 0
    ))

    # Choose behavior
    behavior = GeneralDoublePendulumBehavior(behavior)

    # Setup solvers
    time_evolver = ODEINTTimeEvolver()
    simulation = DoublePendulumSimulation(pendulum, behavior, time_evolver)

    # Simulation parameters
    dt      = 1.0 / 50
    draw_dt = 1.0 / 50

    # Run animation
    animator = DoublePendulumAnimator(simulation)
    animator.init()
    animator.run(dt, draw_dt, 1000)

if __name__ == "__main__":
    run(pi/10, pi/10, FixedQPotential())