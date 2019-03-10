from abc import ABC, abstractmethod
from typing import List, Tuple, Callable

from math import sin, cos, pi, inf
import scipy.constants

###################################################################################################################################################################################
# UTILITY FUNCTIONS
###################################################################################################################################################################################

# Converts an angle in radians to the equivalent angle in radians constrained between -pi and pi, where 0 remains at angle 0
def neg_pi_to_pi(theta: float) -> float:
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

    # Immutable class representing the current state of a double pendulum.
    # See the DoublePendulum class description for information about each parameter.
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
        
        def __str__(self):
                return str([self.theta1(), self.theta2(), self.q(), self.theta1_dot(), self.theta2_dot(), self.q_dot()])
    
    # Immutable class representing the mechanical properties of a double pendulum.
    class Properties:
        # Initializes a new set of properties with the provided values
        #
        #   L             =   length of each arm
        #   m             =   mass of each arm
        #   d             =   moment of inertia factor such that I = 1/2 * m * d^2
        def __init__(self, L: float = 1.0, m: float = 1.0, d: float = 1.0):
            self.__L = L
            self.__m = m
            self.__d = d
        
        def L(self) -> float: return self.__L
        def m(self) -> float: return self.__m
        def d(self) -> float: return self.__d

    # Initializes a new DoublePendulum with the provided properties and initial state
    #
    #   prop          =   mechanical properties of the pendulum
    #   init_state    =   initial state of the pendulum
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
        # Constrain the angles to between -pi and pi for easy interpretation
        # adj_state = DoublePendulum.State(
        #     neg_pi_to_pi(new_state.theta1()),
        #     neg_pi_to_pi(new_state.theta2()),
        #     new_state.q(),
        #     new_state.theta1_dot(),
        #     new_state.theta2_dot(),
        #     new_state.q_dot()
        # )
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
    # def energy_potential_grav(self) -> float:
    #     # Local copies of the state variables for convenience
    #     L = self.prop().L()
    #     m = self.prop().m()
    #     d = self.prop().d()
    #     theta1 = self.state().theta1()
    #     theta2 = self.state().theta2()
        
    #     # Gravitational potential energy of each link:
    #     U_1 = 3/2*m*g*L*cos(theta1)
    #     U_2 = 1/2*m*g*L*cos(theta2)

    #     # Total gravitational potential energy:
    #     U_grav = U_1 + U_2

    #     return U_grav

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
        # T = 1/2*m * theta1_dot**2 * (L**2 + d**2)

        return T

###################################################################################################################################################################################
# IMPLEMENTATION BASE CLASSES
###################################################################################################################################################################################

# Abstract Base Class for representing a potential
class Potential(ABC):

    # Returns the value of the potential given the current state of the pendulum
    # NOTE: Although the state includes the time derivatives of each coordinate,
    #       the potential should be purely a function of the coordinates
    #
    # The potential should be representable as a sum of functions of each individual coordinate
    # The return value of this method should be the value of each of those functions in a vector as follows:
    #    _           _
    #   |  U(theta1)  |
    #   |  U(theta2)  |
    #   |_ U(q)      _|
    #
    # where U_total = U(theta1), U(theta2), U(q)
    #
    @abstractmethod
    def U(self, state: DoublePendulum.State, prop: DoublePendulum.Properties) -> List[float]:
        pass
    
    # Returns the value of the derivative of the potential with respect to each coordinate
    # given the current state of the pendulum
    # 
    # The return value should be a vector in the following form:
    #    _                    _
    #   |  dU(theta1)/dtheta1  |
    #   |  dU(theta2)/dtheta2  |
    #   |_ dU(q)/dq           _|
    #
    @abstractmethod
    def dU_dcoord(self, state: DoublePendulum.State, prop: DoublePendulum.Properties) -> float:
        pass

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
    def state_to_y(self, state: DoublePendulum.State, prop: DoublePendulum.Properties) -> List[float]:
        pass
    
    # Converts an internal state representation (y) back into a DoublePendulum.State
    @abstractmethod
    def y_to_state(self, y: List[float], prop: DoublePendulum.Properties) -> DoublePendulum.State:
        pass
    
    # Core mathematical description of this behavior
    #
    # This is the function f(t, y) mentioned in the DoublePendulumBehavior class description:
    # 
    #   Given an internal state representation y, computes the time derivative, dy/dt
    #
    @abstractmethod
    def dy_dt(self, t: float, y: List[float], prop: DoublePendulum.Properties) -> List[float]:
        pass
    
    # TODO document
    @abstractmethod
    def energy_potential(self, state: DoublePendulum.State, prop: DoublePendulum.Properties):
        pass

# Abstract Base Class for implementing numerical methods to solve the time evolution of a Double Pendulum
class TimeEvolver(ABC):

    def evolve(self, pendulum : DoublePendulum, behavior: DoublePendulumBehavior, dt: float):
        # Convert the current state to y vector
        state_0 = pendulum.state()
        y_0 = behavior.state_to_y(state_0, pendulum.prop())

        # Solve the ODE
        y_1 = self.solve_ode(dt, behavior.dy_dt, y_0, pendulum.prop())

        # Convert resulting y vector back to state
        state_1 = behavior.y_to_state(y_1, pendulum.prop())

        # Update the pendulum
        pendulum.set_state(state_1)
    
    # TODO document
    @abstractmethod
    def solve_ode(self, dt: float, dy_dt: Callable[[float, List[float], DoublePendulum.Properties], List[float]], y_0: List[float], prop: DoublePendulum.Properties):
        pass

###################################################################################################################################################################################
# TIME EVOLVER IMPLEMENTATIONS
###################################################################################################################################################################################

import numpy as np
from scipy.integrate import odeint

# TimeEvolver implementation that uses scipy.integrate.odeint to solve ODEs
class ODEINTTimeEvolver(TimeEvolver):
    def solve_ode(self, dt: float, dy_dt: Callable[[float, List[float], DoublePendulum.Properties], List[float]], y_0: List[float], prop: DoublePendulum.Properties):
        return odeint(dy_dt, y_0, [0, dt], args = (prop,), tfirst = True)[1]

###################################################################################################################################################################################
# POTENTIALS
###################################################################################################################################################################################

class ZeroPotential(Potential):

    def U(self, state: DoublePendulum.State, prop: DoublePendulum.Properties) -> List[float]:
        return [0,0,0]
    
    def dU_dcoord(self, state: DoublePendulum.State, prop: DoublePendulum.Properties) -> List[float]:
        return [0,0,0]

class GravitationalPotential(Potential):
    # Fundamental constants:
    g = scipy.constants.g

    def U(self, state: DoublePendulum.State, prop: DoublePendulum.Properties) -> List[float]:
        g = GravitationalPotential.g

        # Local copies of the state variables for convenience
        L = prop.L()
        m = prop.m()
        theta1 = state.theta1()
        theta2 = state.theta2()
        
        # Gravitational potential energy for each coordinate:
        U_theta1 = 3/2*m*g*L*cos(theta1)
        U_theta2 = 1/2*m*g*L*cos(theta2)
        U_q = 0 # By definition (the pivot point is potential 0)

        return [U_theta1, U_theta2, U_q]
    
    def dU_dcoord(self, state: DoublePendulum.State, prop: DoublePendulum.Properties) -> List[float]:
        g = GravitationalPotential.g

        # Local copies of the state variables for convenience
        L = prop.L()
        m = prop.m()
        theta1 = state.theta1()
        theta2 = state.theta2()

        dU_dtheta1 = -3*g*L*sin(theta1)
        dU_dtheta2 = -1*g*L*sin(theta2)
        dU_dq = 0 # q has no effect on gravitational potential

        return [dU_dtheta1, dU_dtheta2, dU_dq]

# TODO document
class FixedQPotential(Potential):

    def __init__(self, q: float = 0):
        self.__q = q

    def U(self, state: DoublePendulum.State, prop: DoublePendulum.Properties) -> List[float]:
        U_theta1 = 0
        U_theta2 = 0
        U_q = 1/2 * 1E1 * (state.q() - self.__q)**2
        
        return [U_theta1, U_theta2, U_q]
    
    def dU_dcoord(self, state: DoublePendulum.State, prop: DoublePendulum.Properties) -> List[float]:
        dU_dtheta1 = 0
        dU_dtheta2 = 0
        dU_dq = 1E1 * (state.q() - self.__q)
        
        return [dU_dtheta1, dU_dtheta2, dU_dq]

###################################################################################################################################################################################
# BEHAVIOR IMPLEMENTATIONS
###################################################################################################################################################################################

# TODO document
class BaseDoublePendulumBehavior(DoublePendulumBehavior):

    __gravity: Potential = GravitationalPotential()
    def gravity(self) -> Potential:
        return BaseDoublePendulumBehavior.__gravity
    
    def energy_potential(self, state: DoublePendulum.State, prop: DoublePendulum.Properties):
        return np.sum(self.gravity().U(state, prop))

# Implementation of a DoublePendulumBehavior that acts as a single fixed pendulum:
#
# Single:
#   => theta1     = theta2      = theta      (we'll use theta to refer to either angle)
#   => theta1_dot = theta2_dot  = theta_dot  (we'll use theta_dot to refer to either angular velocity)
#
# Fixed:
#   => q = 0
#   => q_dot = 0
#
# The single fixed pendulum is easiest to solve in the following state space (y vector):
#   [
#     theta,
#     theta_dot
#   ]
#
class SingleFixedPendulumBehavior(BaseDoublePendulumBehavior):

    def state_to_y(self, state: DoublePendulum.State, prop: DoublePendulum.Properties) -> List[float]:
        # Verify that the current state is valid for a single pendulum
        # (i.e. angles and angular velocities must be the same for both arms)
        # TODO check for equivalent angles
        # assert (state.theta1() == state.theta2())
        # assert (state.theta1_dot() == state.theta2_dot())
        assert (state.q() == 0)
        assert (state.q_dot() == 0)

        # Construct y vector
        return [state.theta1(), state.theta1_dot()]
    
    def y_to_state(self, y: List[float], prop: DoublePendulum.Properties) -> DoublePendulum.State:
        return DoublePendulum.State(
            theta1     = y[0],
            theta2     = y[0],
            q          = 0,
            theta1_dot = y[1],
            theta2_dot = y[1],
            q_dot      = 0
        )
    
    def dy_dt(self, t: float, y: List[float], prop: DoublePendulum.Properties) -> List[float]:
        L = prop.L() * 2 # We're treating a Double Pendulum like a single one, so L -> 2L
        d = prop.d() * 2 # We're treating a Double Pendulum like a single one, so d -> 2d (since d is proportional to L)

        theta = y[0]
        theta_dot = y[1]

        theta_dot_dot = scipy.constants.g * L / (2 * (L**2/4 + d**2)) * sin(theta) # TODO make more like double pendulum

        return [theta_dot, theta_dot_dot]

# Implementation of DoublePendulumBehavior that acts as a regular double pendulum with a pivot point free to move in the horizontal direction.
#
#   => theta1 and theta2 are independent
#   => theta1_dot and theta2_dot are independent
#   => q is not noecessarily 0
#   => q_dot is not necessarily 0
#
# The double pendulum is easiest to solve in the following state space (y vector):
#   [
#     theta1,
#     theta2,
#     q,
#     p_theta1,
#     p_theta2,
#     p_q
#   ]
#
#   where p_theta1 is the generalized momentum for theta1 (obtained from the Lagrangian)
#         p_theta2 is the generalized momentum for theta2 (obtained from the Lagrangian)
#         p_q      is the generalized momentum for q      (obtained from the Lagrangian)
#
# Note: The factor of 1/2*m that is present in all generalized momenta terms is ignored here since it is constant
#       and does not affect the differential equations
#
# See https://en.wikipedia.org/wiki/Double_pendulum for more information
#
class ForcedDoublePendulumBehavior(BaseDoublePendulumBehavior):
    
    # TODO docs
    def __init__(self, forcing_potential: Potential):
        self.__forcing_potential = forcing_potential

    # There is a set of 3 linear equations (3 knowns, 3 unknowns) that relate the following four quantities:
    #
    #   theta1_dot
    #   theta2_dot
    #   q_dot
    #   p_theta1
    #   p_theta2
    #   p_q
    #
    # The coefficients for these equations are functions of theta1, theta2, L and d and are returned by this method
    # in the form of a 2x2 matrix, A, that satsifies the following equation
    #  _        _   _            _         _          _
    # |          | |  theta1_dot  |   =   |  p_theta1  |
    # |    A     | |  theta2_dot  |       |  p_theta2  |
    # |_        _| |_ q_dot      _|       |_ p_q      _|
    #
    def __coord_dot_p_coord_matrix(self, theta1: float, theta2: float, L: float, m: float, d: float) -> List[List[float]]:
        return np.array([
            [ L**2*5/2 + 2*d**2         , L**2*cos(theta1 - theta2) , 3*L*cos(theta1) ],
            [ L**2*cos(theta1 - theta2) , L**2*1/2 + 2*d**2         , L*cos(theta2)   ],
            [ 3*L*cos(theta1)           , L*cos(theta2)             , 4               ]
        ]) * 1/2*m

    def __p_coord_coord_dot_matrix(self, theta1: float, theta2:float, L: float, m: float, d: float) -> List[List[float]]:
        return np.linalg.inv(self.__coord_dot_p_coord_matrix(theta1, theta2, L, m, d))

    # Transforms a vector of
    #    _            _ 
    #   |  theta1_dot  |
    #   |  theta2_dot  |
    #   |_ q_dot      _|
    #
    # to a vector
    #    _          _
    #   |  p_theta1  |
    #   |  p_theta2  |
    #   |_ p_q      _|
    #
    # using theta1, theta2, L and d
    #
    def __coord_dot_to_p_coord(self, coord_dot: List[float], theta1: float, theta2: float, L: float, m: float, d: float) -> List[float]:
        matrix = self.__coord_dot_p_coord_matrix(theta1, theta2, L, m, d)
        return np.matmul(matrix, coord_dot)

    # Transforms a vector of
    #    _          _ 
    #   |  p_theta1  |
    #   |  p_theta2  |
    #   |_ p_q      _|
    #
    # to a vector
    #    _            _
    #   |  theta1_dot  |
    #   |  theta2_dot  |
    #   |_ q_dot      _|
    #
    # using theta1, theta2, L and d
    #
    def __p_coord_to_coord_dot(self, p_coord: List[float], theta1: float, theta2: float, L: float, m: float, d: float) -> List[float]:
        matrix = self.__p_coord_coord_dot_matrix(theta1, theta2, L, m, d)
        return np.matmul(matrix, p_coord)

    def state_to_y(self, state: DoublePendulum.State, prop: DoublePendulum.Properties) -> List[float]:
        # Construct y vector
        L = prop.L()
        m = prop.m()
        d = prop.d()

        theta1 = state.theta1()
        theta2 = state.theta2()
        q      = state.q()
        theta1_dot = state.theta1_dot()
        theta2_dot = state.theta2_dot()
        q_dot      = state.q_dot()

        p_coord = self.__coord_dot_to_p_coord([theta1_dot, theta2_dot, q_dot], theta1, theta2, L, m, d)

        return [theta1, theta2, q, p_coord[0], p_coord[1], p_coord[2]]
    
    def y_to_state(self, y: List[float], prop: DoublePendulum.Properties) -> DoublePendulum.State:
        L = prop.L()
        m = prop.m()
        d = prop.d()

        theta1 = y[0]
        theta2 = y[1]
        q      = y[2]
        p_theta1 = y[3]
        p_theta2 = y[4]
        p_q      = y[5]

        coord_dot = self.__p_coord_to_coord_dot([p_theta1, p_theta2, p_q], theta1, theta2, L, m, d)

        return DoublePendulum.State(
            theta1     = theta1,
            theta2     = theta2,
            q          = q,
            theta1_dot = coord_dot[0],
            theta2_dot = coord_dot[1],
            q_dot      = coord_dot[2]
        )
    
    def dy_dt(self, t: float, y: List[float], prop: DoublePendulum.Properties) -> List[float]:
        L = prop.L()
        d = prop.d()

        # Local variables for the y vector elements
        theta1 = y[0]
        theta2 = y[1]
        q      = y[2]
        p_theta1 = y[3]
        p_theta2 = y[4]
        p_q      = y[5]

        # Calculate the time derivatives of each coordinate from the generalized momenta
        coord_dot = self.__p_coord_to_coord_dot([p_theta1, p_theta2, p_q], theta1, theta2, L, m, d)
        theta1_dot = coord_dot[0]
        theta2_dot = coord_dot[1]
        q_dot      = coord_dot[2]

        # Store everything is a state instance for passing to functions
        state = DoublePendulum.State(theta1, theta2, q, theta1_dot, theta2_dot, q_dot)

        # Calculate the time derivatives of the generalized momenta
        gravity_terms = self.gravity().dU_dcoord(state, prop)
        forcing_terms = self.__forcing_potential.dU_dcoord(state, prop)

        p_theta1_dot = 1/2*m * (-3*q_dot*L*theta1_dot*sin(theta1) - L**2*theta1_dot*theta2_dot*sin(theta1 - theta2)) - (gravity_terms[0] + forcing_terms[0])
        p_theta2_dot = 1/2*m * (-1*q_dot*L*theta2_dot*sin(theta2) + L**2*theta1_dot*theta2_dot*sin(theta1 - theta2)) - (gravity_terms[1] + forcing_terms[1])
        p_q_dot      = gravity_terms[2] + forcing_terms[2]

        return [theta1_dot, theta2_dot, q_dot, p_theta1_dot, p_theta2_dot, p_q_dot]
    
    def energy_potential(self, state: DoublePendulum.State, prop: DoublePendulum.Properties):
        return np.sum(self.gravity().U(state, prop) + self.__forcing_potential.U(state, prop))

###################################################################################################################################################################################
# PENDULATION SIMULATION
###################################################################################################################################################################################

# Class that manages the evolution of the double pendulum over time
class DoublePendulumSimulation:
    def __init__(self, pendulum: DoublePendulum, behavior: DoublePendulumBehavior, time_evolver: TimeEvolver):
        self.__pendulum = pendulum
        self.__behavior = behavior
        self.__time_evolver = time_evolver

        self.__elapsed_time = 0

    def pendulum(self) -> DoublePendulum: return self.__pendulum
    def behavior(self) -> DoublePendulumBehavior: return self.__behavior
    def time_evolver(self) -> TimeEvolver: return self.__time_evolver
    
    def elapsed_time(self) -> float: return self.__elapsed_time

    def step(self, dt: float):
        self.time_evolver().evolve(self.pendulum(), self.behavior(), dt)
        self.__elapsed_time += dt

    def step_until(self, dt: float, t_final: float):
        while (self.elapsed_time() < t_final):
            self.step(dt)
    
    def step_for(self, dt: float, delta_t: float):
        local_elapsed_time = 0.0
        while (local_elapsed_time < delta_t):
            self.step(dt)
            local_elapsed_time += dt

###################################################################################################################################################################################
# ANIMATORS
###################################################################################################################################################################################

from time import time

import matplotlib.pyplot as plt
import matplotlib.animation as animation

# TODO class documentation
class DoublePendulumAnimator:
    def __init__(self, simulation: DoublePendulumSimulation):
        self.__simulation = simulation

    # TODO documentation
    def init(self):
        # Pendulum subplot
        self.fig = plt.figure(figsize=(8, 8))
        scale_margin_factor_x = 6
        scale_margin_factor_y = 3
        L = self.__simulation.pendulum().prop().L()
        scale_x = (-1 * scale_margin_factor_x * L, scale_margin_factor_x * L)
        scale_y = (-1 * scale_margin_factor_y * L, scale_margin_factor_y * L)
        self.ax_main = self.fig.add_subplot(211, aspect='equal', autoscale_on=False, xlim=scale_x, ylim=scale_y)
        self.ax_main.grid()

        # Lines
        self.line_main, = self.ax_main.plot([], [], '-', lw=4)
        self.time_text_main = self.ax_main.text(0.02, 0.95, '', transform=self.ax_main.transAxes)
        self.energy_text = self.ax_main.text(0.02, 0.90, '', transform=self.ax_main.transAxes)

        # self.ax_q = self.fig.add_subplot(212, autoscale_on=True)
        # self.ax_q.set_xlabel('Time (seconds)')
        # self.ax_q.set_ylabel('q (metres)')
        # self.ax_q.grid()
        
        # self.line_q, = self.ax_q.plot([], [])

        self.__reset()

    # TODO documentation
    def __reset(self):
        # TODO reset simulation

        self.line_main.set_data([],[])
        self.time_text_main.set_text('')
        self.energy_text.set_text('')

        #self.line_q.set_data([],[])

        # Required for matplotlib to update
        return self.line_main, self.time_text_main, self.energy_text#, self.line_q

    # Internal function that performs a single animation step
    def __animate(self, i: int, dt: float, draw_dt: float):
        # Simulate next step
        self.__simulation.step_for(dt, draw_dt)

        # Update pendulum position plot
        ((x_0, y_0), (x_1, y_1), (x_2, y_2)) = self.__simulation.pendulum().position_ends()
        x = [x_0, x_1, x_2]
        y = [y_0, y_1, y_2]
        self.line_main.set_data(x, y)

        # Update elapsed time text
        self.time_text_main.set_text('time = %.1f s' % self.__simulation.elapsed_time())

        # Update energy text
        potential = self.__simulation.behavior().energy_potential(self.__simulation.pendulum().state(), self.__simulation.pendulum().prop())
        kinetic = self.__simulation.pendulum().energy_kinetic()
        total_energy = potential + kinetic
        self.energy_text.set_text('potential = %.3f, kinetic = %.3f, total = %.3f' % (potential, kinetic, total_energy))

        # # Update q plot
        # if (self.plot_q is True):
        #     self.line_q.set_data(self.t, self.q)
        #     self.ax_q.relim()
        #     self.ax_q.autoscale_view(True,True,True)

        # Required for matplotlib to update
        return self.line_main, self.time_text_main, self.energy_text#, self.line_q

    # Runs and displays an animation of the pendulum
    #
    #   dt       = time step for the simulation (seconds)
    #   draw_dt  = time between animation frame updates (seconds)
    #   t_final  = time at which the simulation will stop (seconds)
    #
    def run(self, dt: float, draw_dt: float, t_final: float):
        interval = draw_dt * 1000 # interval is in milliseconds
        frames = int(t_final / dt)

        self.ani = animation.FuncAnimation(self.fig, self.__animate, fargs = (dt, draw_dt), frames=frames, interval=interval, blit=True, init_func=self.__reset, repeat=False)

        plt.show()

###################################################################################################################################################################################
# ANIMATORS
###################################################################################################################################################################################

from math import sqrt

if __name__ == "__main__":

    # Setup pendulum
    L = 1            # m
    m = 2            # kg
    d = sqrt(1/12)*L # m

    pendulum = DoublePendulum(DoublePendulum.Properties(L, m, d), DoublePendulum.State(
        theta1     = pi/2,
        theta2     = pi/2,
        q          = 0,
        theta1_dot = 0,
        theta2_dot = 0,
        q_dot      = 0
    ))

    # Choose behavior
    #behavior = SingleFixedPendulumBehavior()
    behavior = ForcedDoublePendulumBehavior(ZeroPotential())

    # Setup solvers
    time_evolver = ODEINTTimeEvolver()
    simulation = DoublePendulumSimulation(pendulum, behavior, time_evolver)

    # Simulation parameters
    dt = 1.0 / 50
    draw_dt = 1/50

    # Run animation
    animator = DoublePendulumAnimator(simulation)
    animator.init()
    animator.run(dt, draw_dt, 1000)