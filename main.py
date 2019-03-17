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

class TestPotential(BasePotential):
    def __init__(self, k1: float = 0, k2: float = 0, k3: float = 0, k4: float = 0):
        self.__k1 = k1
        self.__k2 = k2
        self.__k3 = k3
        self.__k4 = k4

    def U(self, t: float, state: DoublePendulum.State, prop: DoublePendulum.Properties) -> List[float]:
        return [
            0,
            0,
            -1*(self.__k1 * state.theta1() + self.__k2 * state.theta1_dot() + self.__k3 * state.theta2() + self.__k4 * state.theta2_dot()) * state.q()
        ]
    
    def dU_dcoord(self, t: float, state: DoublePendulum.State, prop: DoublePendulum.Properties) -> List[float]:
        return [
            0,
            0,
            -1*(self.__k1 * state.theta1() + self.__k2 * state.theta1_dot() + self.__k3 * state.theta2() + self.__k4 * state.theta2_dot())
        ]

def run(pendulum: DoublePendulum, behavior: DoublePendulumBehavior):
    # Setup solvers
    time_evolver = ODEINTTimeEvolver()
    simulation = DoublePendulumSimulation(pendulum, behavior, time_evolver)

    # Simulation parameters
    dt      = 1.0 / 100
    draw_dt = 1.0 / 100

    # Run animated simulation
    animator = DoublePendulumAnimator(simulation)
    animator.init()
    animator.run(dt, draw_dt, 1000)

if __name__ == "__main__":
    # run_potential(theta1 = pi/100, theta2 = pi/100, q = 0, potential = FixedQPotential())
    # run_forcing(theta1 = pi/100, theta2 = pi/100, forcing_function = SinusoidalForcing(amplitude = -0.2, frequency = 3), potential = SinglePendulumPotential())
    run_potential(theta1 = pi/10, theta2 = pi/10, potential = TestPotential(k1 = 2E3, k2 = 10))