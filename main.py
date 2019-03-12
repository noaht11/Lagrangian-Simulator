from double_pendulum import *

from math import sin, cos, sqrt

def setup_pendulum(theta1: float = 0, theta2: float = 0, q: float = 0, theta1_dot: float = 0, theta2_dot: float = 0, q_dot: 0 = float, L: float = 1, m: float = 2):
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

def run_forcing(theta1: float = 0, theta2: float = 0, q: float = 0, q_dot: float = 0, forcing_function: QForcingFunction = FixedQForcingFunction(), potential: Potential = ZeroPotential()):
    run(
        setup_pendulum(theta1 = theta1, theta2 = theta2, q = q, q_dot = q_dot),
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

    
class SinusoidalForcing(QForcingFunction):
    
    def __init__(self, amplitude: float = 1, frequency: float = 0):
        self.__amplitude = amplitude
        self.__frequency = frequency

    def q(self, t: float) -> float: return self.__amplitude * sin(2*pi*self.__frequency*t)
    def dq_dt(self, t: float) -> float: return self.__amplitude * 2*pi*self.__frequency * cos(2*pi*self.__frequency*t)

if __name__ == "__main__":
    #run_potential(theta1 = pi/100, theta2 = pi/100, q = 0, potential = FixedQPotential())
    forcing = SinusoidalForcing(amplitude = 0.2, frequency = 5)
    run_forcing(theta1 = pi/100, theta2 = pi/100, q = forcing.q(0), q_dot = forcing.dq_dt(0), forcing_function = forcing)