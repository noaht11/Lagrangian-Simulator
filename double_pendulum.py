from typing import List, Tuple

###################################################################################################################################################################################
# UTILITY FUNCTIONS
###################################################################################################################################################################################

from math import pi

# Converts an angle in radians to the equivalent angle in radians constrained between -pi and pi, where 0 remains at angle 0
def __neg_pi_to_pi(theta: float) -> float:
    modded = theta % (2*pi)
    return modded + (modded > pi) * (-2*pi)

###################################################################################################################################################################################
# DOUBLE PENDULUM DEFINITION
###################################################################################################################################################################################

from math import sin, cos
from scipy.constants import g

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

    # L             =   length of each arm
    # m             =   mass of each arm
    # d             =   moment of inertia factor such that I = 1/2 * m * d^2
    # init_state    =   initial state of the pendulum in the form described in the class description
    #
    def __init__(self, L: float = 1.0, m: float = 1.0, d: float = 1.0, init_state: State = State()):
        self.L = L
        self.m = m
        self.d = d

        self.__state = init_state
    
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
        L = self.L
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
        L = self.L
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
        L = self.L
        m = self.m
        d = self.d
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
        L = self.L
        m = self.m
        d = self.d
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

from abc import ABC, abstractmethod

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
    def get_y(self, pendulum_state: DoublePendulum.State) -> List[float]:
        pass
    
    # Converts an internal state representation (y) back into a DoublePendulum.State
    @abstractmethod
    def get_pendulum_state(self, y: List[float]) -> DoublePendulum.State:
        pass
    
    # Core mathematical description of this behavior
    #
    # This is the function f(t, y) mentioned in the DoublePendulumBehavior class description:
    # 
    #   Given an internal state representation y, computes the time derivative, dy/dt
    #
    @abstractmethod
    def dstate_dt(self, t: float, y: List[float]) -> List[float]:
        pass

# Abstract Base Class for implementing numerical methods to solve the time evolution of a Double Pendulum
class TimeEvolver(ABC):

    @abstractmethod
    def evolve(self, pendulum : DoublePendulum, behavior: DoublePendulumBehavior, dt: float):
        pass

###################################################################################################################################################################################
# BEHAVIOR IMPLEMENTATIONS
###################################################################################################################################################################################

class SinglePendulum:
    pass

# from time import time
# import numpy as np
# from scipy.integrate import solve_ivp
# from scipy.integrate import odeint
# from scipy.integrate import RK45

# import matplotlib.pyplot as plt
# import matplotlib.animation as animation