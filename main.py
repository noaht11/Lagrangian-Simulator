from double_pendulum import *

from math import sin, cos, sqrt

class HarmonicOscillator(Potential):

    def __init__(self, frequency: float):
        self.__w = frequency

    def U(self, state: DoublePendulum.State, prop: DoublePendulum.Properties) -> List[float]:
        return [0, 0, 1/2*(self.__w**2)*(state.q()**4)]
    
    def dU_dcoord(self, state: DoublePendulum.State, prop: DoublePendulum.Properties) -> List[float]:
        return [0, 0, 2*(self.__w**2)*state.q()**3]

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
    run(-pi/100, pi/100, HarmonicOscillator(50))