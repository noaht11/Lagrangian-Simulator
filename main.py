from math import sin, cos, sqrt, exp
import numpy as np
import scipy.constants

from pendulum.core import *
from pendulum.behavior import *
from pendulum.potential import *
from pendulum.simulate import *
from pendulum.animate import *

###################################################################################################################################################################################
# RUN UTILITIES
###################################################################################################################################################################################

def setup_pendulum(theta1: float = 0, theta2: float = 0, q: float = 0, theta1_dot: float = 0, theta2_dot: float = 0, q_dot: float = 0, L: float = 0.1, m: float = 0.1, d: float = 1, **kwargs):
    pendulum = DoublePendulum(DoublePendulum.Properties(L, m, d, **kwargs), DoublePendulum.State(
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

def run(pendulum: DoublePendulum, behavior: DoublePendulumBehavior, var_trackers: List[VariableTracker] = []):
    # Setup solvers
    time_evolver = ODEINTTimeEvolver()
    simulation = DoublePendulumSimulation(pendulum, behavior, time_evolver, var_trackers)

    # Simulation parameters
    sim_dt  = 1.0 / 50
    draw_dt = 1.0 / 50

    # Run animated simulation
    animator = DoublePendulumAnimator(simulation)
    animator.init()
    animator.run(sim_dt, draw_dt, 1000)

def run_fft_q(pendulum: DoublePendulum, behavior: DoublePendulumBehavior, end_time: float):
    # Variable tracker
    q_tracker = VariableTracker(pendulum.state().q(), lambda t, state, prop, behavior: state.q())

    # Setup solvers
    time_evolver = ODEINTTimeEvolver()
    simulation = DoublePendulumSimulation(pendulum, behavior, time_evolver, [q_tracker])

    # Simulation parameters
    sim_dt  = 1.0 / 50

    # Run simulation
    simulation.step_for(sim_dt, end_time)

    # Take FFT
    fft_result = np.fft.fft(q_tracker.data)
    fft_mags = np.absolute(fft_result)

    # Corresponding frequencies
    freq = np.fft.fftfreq(simulation.t_tracker().data.shape[-1])

    # Plot the original function and FFT
    plt.figure()
    plt.plot(simulation.t_tracker().data, q_tracker.data)

    plt.figure()
    plt.plot(freq, fft_mags)

    plt.show()


###################################################################################################################################################################################
# TEST POTENTIALS
###################################################################################################################################################################################

class TestPotential1(BasePotential):
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

def test_force_1(t: float, prop: DoublePendulum.Properties, w: float, B: float, d_converter: Callable[[DoublePendulum.Properties], float]):
    g = scipy.constants.g

    L = prop.L() * 2
    m = prop.m() * 2

    I = m * d_converter(prop)**2
    I_R = I + m * (L/2)**2

    return -1*w**2*B/(L/2) * (1 + I_R * w**2 / (m*g*(L/2))) * cos(w*t)

def test_force_2(t: float, prop: DoublePendulum.Properties, A: float, w_c: float, w_m: float, phi: float):
    return A * -1*exp(-1*sin(w_c*t - phi)) * (w_c*cos(w_c*t)*cos(w_m*t - phi) + w_m*sin(w_m*t))

###################################################################################################################################################################################
# MAIN
###################################################################################################################################################################################

def I_cylinder(L, R, m):
    return 1/4*m*R**2 + 1/12*m*L**2

def d_cyclinder(L, R, m):
    I = I_cylinder(L, R, m)
    return sqrt(I/m)

def d_converter_cylinder(prop):
    L = prop.L() * 2
    R = prop.extras()["R"]
    m = prop.m() * 2

    return d_cyclinder(L, R, m)

if __name__ == "__main__":
    # Universal constants
    g = scipy.constants.g

    # Pendulum properties
    L = 0.20
    R = 0.065
    m = 0.072
    d = d_cyclinder(L, R, m)

    # Forcing parameters
    theta_0 = pi/20
    B = -5
    w = sqrt(-1*theta_0*m*g*(L/2)/B)

    # force = lambda t, prop: test_force_1(t, prop, w, B, d_converter_cylinder)
    force = lambda t, prop: test_force_2(t, prop, -0.01*0, 3, 50, -3*pi/2)
    forcing_potential = ForceQPotential(force)

    # variable trackers
    force_tracker = VariableTracker(0, lambda t, state, prop, behavior: force(t, prop))
    q_tracker = VariableTracker(0, lambda t, state, prop, behavior: state.q())

    # Pendulum and Behavior
    pendulum = setup_pendulum(theta1 = theta_0, theta2 = theta_0, L = L, m = m, d = d, R = R)
    behavior = GeneralSinglePendulumBehavior(forcing_potential = forcing_potential, d_converter = d_converter_cylinder)

    # Run FFT
    run_fft_q(
        pendulum,
        behavior,
        10
    )

    # Run animation
    # run(
    #     pendulum,
    #     behavior,
    #     [q_tracker]
    # )


    # run_potential(theta1 = pi/10, theta2 = pi/10, q = 0)
    # run_potential(theta1 = pi/100, theta2 = pi/100, q = 0, potential = FixedQPotential())
    # run_forcing(theta1 = pi/100, theta2 = pi/100, forcing_function = SinusoidalForcing(amplitude = 0.30, frequency = 4, phase = pi/2, damping = 3.3), potential = SinglePendulumPotential())
    # run_potential(theta1 = pi/10, theta2 = pi/10, q = -0.15, potential = TestPotential(k1 = 5, k2 = 0.01) + SinglePendulumPotential())

    # message = SymbolicSinusoidalForcing(A = 0.3, w = 8*sym.pi, phi = sym.pi/2, k = 3.3)
    # carrier = SymbolicSinusoidalForcing(w = 10, phi = sym.pi/2)

    # t = symbols('t')

    # message = SymbolicSinusoidalForcing(A = -0.5, w = 8*sym.pi, phi = sym.pi/2)
    # carrier = SymbolicSinusoidalForcing(w = 1.4, phi = -1*sym.pi/2)
    # multiplier = (t - 0.8)

    # forcing = message * carrier * multiplier
    # sym.pprint(forcing)

    # forcing = SymbolicSinusoidalForcing(A = 0.06, w = 8*sym.pi, phi = sym.pi/2, k = 1.55)

    # run_forcing(theta1 = pi/10, theta2 = pi/10, forcing_function = SymbolicForcing(forcing), potential = SinglePendulumPotential())




    # WORKING:
    # run_potential(theta1 = pi/10, theta2 = pi/10, q = -0.15, potential = TestPotential(k1 = 5, k2 = 0.01) + SinglePendulumPotential())