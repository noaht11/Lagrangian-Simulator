from abc import ABC, abstractmethod
from typing import List, Tuple

from math import sin, cos, pi
from scipy.constants import g

###################################################################################################################################################################################
# UTILITY FUNCTIONS
###################################################################################################################################################################################

# Converts an angle in radians to the equivalent angle in radians constrained between -pi and pi, where 0 remains at angle 0
def __neg_pi_to_pi(theta: float) -> float:
    modded = theta % (2*pi)
    return modded + (modded > pi) * (-2*pi)

###################################################################################################################################################################################
# DOUBLE PENDULUM DEFINITION
###################################################################################################################################################################################

# Class representing the instantaneous state of a double pendulum with at most 3 degrees of freedom:
#
#   1.  theta1  (angle between the first arm and the vertical, 0 = straight up from the pivot point)
#   2.  theta2  (angle between the second arm and the vertical, 0 = straight up from the joint)
#   3.  q       (horizontal position of the pivot point relative to an arbitrary origin)
#
# The state of the double pendulum is contained in an array of six values (in the following order):
#
#   [0]    =   theta1
#   [1]    =   theta2
#   [2]    =   q
#   [3]    =   theta1_dot   (time derivative of theta1)
#   [4]    =   theta2_dot   (time derivative of theta2)
#   [5]    =   q_dot        (time derivative of q)
#
class DoublePendulum:

    class State:
        def __init__(self, theta1: float = 0, theta2: float = 0, q: float = 0, theta1_dot: float = 0, theta2_dot: float = 0, q_dot: float = 0):
            self.__theta1     = theta1
            self.__theta2     = theta2
            self.__q          = q
            self.__theta1_dot = theta1_dot
            self.__theta2_dot = theta2_dot
            self.__q_dot      = q_dot
        
        def theta1(self)     -> float : return self.__theta1
        def theta2(self)     -> float : return self.__theta2
        def q(self)          -> float : return self.__q
        def theta1_dot(self) -> float : return self.__theta1_dot
        def theta2_dot(self) -> float : return self.__theta2_dot
        def q_dot(self)      -> float : return self.__q_dot
    
    class Properties:
        # L             =   length of each arm
        # m             =   mass of each arm
        # d             =   moment of inertia factor such that I = 1/2 * m * d^2
        def __init__(self, L: float = 1.0, m: float = 1.0, d: float = 1.0):
            self.__L = L
            self.__m = m
            self.__d = d
        
        def L(self) -> float: return self.__L
        def m(self) -> float: return self.__m
        def d(self) -> float: return self.__d

    # prop          =   mechanical properties of the pendulum
    # init_state    =   initial state of the pendulum
    def __init__(self, prop: Properties = Properties(), init_state: State = State()):
        self.__prop = prop
        self.__state = init_state
    
    # Returns the mechanical properties of the pendulum
    def prop(self) -> Properties:
        return self.__prop

    # Returns the current state
    def state(self) -> State:
        return self.__state

    # Sets the current state to new_state
    def set_state(self, new_state: State):
        self.__state = new_state

    # Calculates and returns the positions of the endpoints of the arms of the pendulum
    #
    # The return value is a tuple of the following form:
    #   (
    #       (pivot_x, pivot_y),
    #       (joint_x, joint_y),
    #       (end_x  , end_y  )
    #   )
    def position_ends(self) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        # Local copies of the state variables for convenience
        L = self.prop().L()
        theta1 = self.state().theta1()
        theta2 = self.state().theta2()
        q = self.state().q()
        
        x_1 = q + L*sin(theta1)
        y_1 = L*cos(theta1)

        x_2 = q + L*(sin(theta1) + sin(theta2))
        y_2 = L*(cos(theta1) + cos(theta2))

        return ((q, 0.0), (x_1, y_1), (x_2, y_2))
    
    # Calculates and returns the positions of the pivot point and COM (centre of mass) of each arm of the pendulum
    #
    # The return value is a tuple of the following form:
    #   (
    #       (pivot_x, pivot_y),
    #       (COM_1_x, COM_1_y),
    #       (COM_2_x, COM_2_y)
    #   )
    def position_COMs(self) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        # Local copies of the state variables for convenience
        L = self.prop().L()
        theta1 = self.state().theta1()
        theta2 = self.state().theta2()
        q = self.state().q()

        x_1 = q + L/2*sin(theta1)
        y_1 = L/2*cos(theta1)

        x_2 = q + L*(sin(theta1) + 1/2*sin(theta2))
        y_2 = L*(cos(theta1) + 1/2*cos(theta2))

        return ((q, 0.0), (x_1, y_1), (x_2, y_2))
    
    # Calculates and returns the gravitation potential energy of the current state of the pendulum
    def energy_potential_grav(self) -> float:
        # Local copies of the state variables for convenience
        L = self.prop().L()
        m = self.prop().m()
        d = self.prop().d()
        theta1 = self.state().theta1()
        theta2 = self.state().theta2()
        
        # Gravitational potential energy of each link:
        U_1 = 3/2*m*g*L*cos(theta1)
        U_2 = 1/2*m*g*L*cos(theta2)

        # Total gravitational potential energy:
        U_grav = U_1 + U_2

        return U_grav

    # Calculates and returns the kinetic energy of the current state of the pendulum
    def energy_kinetic(self) -> float:
        # Local copies of the state variables for convenience
        L = self.prop().L()
        m = self.prop().m()
        d = self.prop().d()
        theta1 = self.state().theta1()
        theta2 = self.state().theta2()
        q = self.state().q()
        theta1_dot = self.state().theta1_dot()
        theta2_dot = self.state().theta2_dot()
        q_dot = self.state().q_dot()

        # This is just the final formula for kinetic energy after a lot of re-arranging, so there's no good way to decompose it
        T = 1/2*m * (2*q_dot**2   +   q_dot*L*(3*cos(theta1)*theta1_dot + cos(theta2)*theta2_dot)   +   L**2*(1/4*(5*theta1_dot**2 + theta2_dot**2) + theta1_dot*theta2_dot*cos(theta1 - theta2))   +   d**2*(theta1_dot**2 + theta2_dot**2))

        return T

###################################################################################################################################################################################
# IMPLEMENTATION BASE CLASSES
###################################################################################################################################################################################

# Abstract Base Class for implementations of the behavior of a DoublePendulum
#
# The behavior of a double pendulum is defined by a differential equation of the following form:
#
#   dy/dt = f(t, y)
#
# where y is a vector of values describing the current state of the pendulum
#
# (y does not have to have the exact same data the same as DoublePendulum.State but you must be able to
#  translate from one to the other, i.e. they must effectively contain the same information)
#
class DoublePendulumBehavior(ABC):

    # Converts a DoublePendulum.State to the internal state representation (y) that is used in the differential equation
    # defining this behavior
    #
    # The return value of this method will be passed to dstate_dt
    @abstractmethod
    def state_to_y(self, pendulum_state: DoublePendulum.State) -> List[float]:
        pass
    
    # Converts an internal state representation (y) back into a DoublePendulum.State
    @abstractmethod
    def y_to_state(self, y: List[float]) -> DoublePendulum.State:
        pass
    
    # Core mathematical description of this behavior
    #
    # This is the function f(t, y) mentioned in the DoublePendulumBehavior class description:
    # 
    #   Given an internal state representation y, computes the time derivative, dy/dt
    #
    @abstractmethod
    def dy_dt(self, t: float, y: List[float]) -> List[float]:
        pass

# Abstract Base Class for implementing numerical methods to solve the time evolution of a Double Pendulum
class TimeEvolver(ABC):

    @abstractmethod
    def evolve(self, pendulum : DoublePendulum, behavior: DoublePendulumBehavior, dt: float):
        pass

###################################################################################################################################################################################
# BEHAVIOR IMPLEMENTATIONS
###################################################################################################################################################################################

class SingleFixedPendulumBehavior:
    def state_to_y(self, pendulum_state: DoublePendulum.State) -> List[float]:
        assert (pendulum_state.theta1() == pendulum_state.theta2())
        assert (pendulum_state.theta1_dot() == pendulum_state.theta2_dot())

        return [pendulum_state.theta1(), pendulum_state.theta1_dot()]
    
    def y_to_state(self, y: List[float]) -> DoublePendulum.State:
        return DoublePendulum.State(
            theta1     = y[0],
            theta2     = y[0],
            q          = 0,
            theta1_dot = y[1],
            theta2_dot = y[1],
            q_dot      = 0
        )
    
    def dy_dt(self, t: float, y: List[float], args: Tuple[DoublePendulum.Properties]) -> List[float]:
        (prop) = args
        L = prop.L() * 2 # We're treating a Double Pendulum like a single one, so L -> 2L
        d = prop.d()

        theta = y[0]
        theta_dot = y[1]

        theta_dot_dot = g * L / (2 * (L**2 + d**2)) * sin(theta)

        return [theta_dot, theta_dot_dot]

###################################################################################################################################################################################
# TIME EVOLVER IMPLEMENTATIONS
###################################################################################################################################################################################

from scipy.integrate import odeint

# TODO class documentation
class ODEINTTimeEvolver:
    def evolve(self, pendulum : DoublePendulum, behavior: DoublePendulumBehavior, dt: float):
        state_0 = pendulum.state()
        y_0 = behavior.state_to_y(state_0)

        y_1 = odeint(behavior.dy_dt, y_0, [0, dt], args = (pendulum.prop()), tfirst = True)[1]
        state_1 = behavior.y_to_state(y_1)

        pendulum.set_state(state_1)

###################################################################################################################################################################################
# PENDULATION SIMULATION
###################################################################################################################################################################################

# TODO class documentation
class DoublePendulumSimulation:
    def __init__(self, pendulum: DoublePendulum, behavior: DoublePendulumBehavior, time_evolver: TimeEvolver):
        self.__pendulum = pendulum
        self.__behavior = behavior
        self.__time_evolver = time_evolver
    
    def pendulum(self) -> DoublePendulum: return self.__pendulum
    def behavior(self) -> DoublePendulumBehavior: return self.__behavior
    def time_evolver(self) -> TimeEvolver: return self.__time_evolver

    def step(self, dt: float):
        self.time_evolver().evolve(self.pendulum(), self.behavior(), dt)
        # TODO save output over time
        # Track time elapsed

###################################################################################################################################################################################
# ANIMATORS
###################################################################################################################################################################################

from time import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# TODO class documentation
class DoublePendulumAnimator:
    def __init__(self, simulation: DoublePendulumSimulation, dt: float, plot_q: bool = False):
        self.__simulation = simulation
        self.dt = dt

        self.plot_q = plot_q

    def init(self):
        self.fig = plt.figure(figsize=(8, 8))
        scale_margin_factor_x = 6
        scale_margin_factor_y = 3
        L = self.__simulation.pendulum().prop().L()
        scale_x = (-1 * scale_margin_factor_x * L, scale_margin_factor_x * L)
        scale_y = (-1 * scale_margin_factor_y * L, scale_margin_factor_y * L)
        self.ax_main = self.fig.add_subplot(211, aspect='equal', autoscale_on=False, xlim=scale_x, ylim=scale_y)
        self.ax_main.grid()

        self.line_main, = self.ax_main.plot([], [], '-', lw=4)
        self.time_text_main = self.ax_main.text(0.02, 0.95, '', transform=self.ax_main.transAxes)
        self.energy_text = self.ax_main.text(0.02, 0.90, '', transform=self.ax_main.transAxes)

        self.ax_q = self.fig.add_subplot(212, autoscale_on=True)
        self.ax_q.set_xlabel('Time (seconds)')
        self.ax_q.set_ylabel('q (metres)')
        self.ax_q.grid()
        
        self.line_q, = self.ax_q.plot([], [])

        self.__reset()

    def __reset(self):
        # TODO reset pendulum
        
        if (self.plot_q is True):
            self.t = np.array([0])

            self.theta1 = np.array(self.__simulation.pendulum().state().theta1())
            self.theta2 = np.array(self.__simulation.pendulum().state().theta2())
            self.q       = np.array(self.__simulation.pendulum().state().q())

        self.line_main.set_data([],[])
        self.time_text_main.set_text('')
        self.energy_text.set_text('')

        self.line_q.set_data([],[])

        # Required for matplotlib to update
        return self.line_main, self.time_text_main, self.energy_text, self.line_q

    def __animate(self, i):
        self.__simulation.step(self.dt)

        if (self.plot_q is True):
            self.t = np.append(self.t, self.t[-1] + self.dt)
            self.q = np.append(self.q, self.__simulation.pendulum().state().q())
        
        ((x_0, y_0), (x_1, y_1), (x_2, y_2)) = self.__simulation.pendulum().position_ends()
        x = [x_0, x_1, x_2]
        y = [y_0, y_1, y_2]
        self.line_main.set_data(x, y)

        #self.time_text_main.set_text('time = %.1f s' % self.pendulum.time_elapsed)

        potential = self.__simulation.pendulum().energy_potential_grav()
        kinetic = self.__simulation.pendulum().energy_kinetic()
        total_energy = potential + kinetic
        self.energy_text.set_text('potential = %.3f, kinetic = %.3f, total = %.3f' % (potential, kinetic, total_energy))

        if (self.plot_q is True):
            self.line_q.set_data(self.t, self.q)
            self.ax_q.relim()
            self.ax_q.autoscale_view(True,True,True)

        # Required for matplotlib to update
        return self.line_main, self.time_text_main, self.energy_text, self.line_q

    def run(self, frames):
        t0 = time()
        self.__animate(0)
        t1 = time()
        interval = (1000 * self.dt - (t1-t0)) / 1

        self.ani = animation.FuncAnimation(self.fig, self.__animate, frames=frames, interval=interval, blit=True, init_func=self.__reset, repeat=False)

        plt.show()

###################################################################################################################################################################################
# ANIMATORS
###################################################################################################################################################################################

from math import sqrt

if __name__ == "__main__":
    L = 2            # m
    m = 1            # kg
    d = sqrt(1/12)*L # m

    pendulum = DoublePendulum(DoublePendulum.Properties(L, m, d), DoublePendulum.State(
        theta1     = 0,
        theta2     = 0,
        q          = 0,
        theta1_dot = 0,
        theta2_dot = 0,
        q_dot      = 0
    ))

    behavior = SingleFixedPendulumBehavior()
    time_evolver = ODEINTTimeEvolver()

    simulation = DoublePendulumSimulation(pendulum, behavior, time_evolver)

    dt = 1.0 / 100

    animator = DoublePendulumAnimator(simulation, dt, plot_q = True)
    animator.init()
    animator.run(10000)