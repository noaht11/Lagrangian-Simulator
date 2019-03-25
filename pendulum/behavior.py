from abc import ABC, abstractmethod
from typing import List, Tuple, Callable

import numpy as np

from pendulum.core import *
from pendulum.potential import *

###################################################################################################################################################################################
# ABSTRACT BASE CLASSES
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
    def state_to_y(self, t: float, state: DoublePendulum.State, prop: DoublePendulum.Properties) -> List[float]:
        pass
    
    # Converts an internal state representation (y) back into a DoublePendulum.State
    @abstractmethod
    def y_to_state(self, t: float, y: List[float], prop: DoublePendulum.Properties) -> DoublePendulum.State:
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
    
    # Returns the potential energy, a pendulum in the given state and with the given properties would have
    # according to this behavior
    @abstractmethod
    def energy_potential(self, t: float, state: DoublePendulum.State, prop: DoublePendulum.Properties) -> float:
        pass

###################################################################################################################################################################################
# BASE BEHAVIOR
###################################################################################################################################################################################

# Base implementation for a DoublePendulumBehavior that provides a convenient way to behave at least according to gravity
class BaseDoublePendulumBehavior(DoublePendulumBehavior):

    # Internal static (class) reference to an instance of GravitationalPotential so we don't have to keep instantiating one
    __gravity: Potential = GravitationalPotential()

    # Returns an instance of a gravitational potential
    def gravity(self) -> Potential:
        return BaseDoublePendulumBehavior.__gravity
    
    # Implementation of energy_potential that just returns the potential energy due to gravity
    def energy_potential(self, t: float, state: DoublePendulum.State, prop: DoublePendulum.Properties) -> float:
        return np.sum(self.gravity().U(t, state, prop))


###################################################################################################################################################################################
# BEHAVIOR IMPLEMENTATIONS
###################################################################################################################################################################################

# Implementation of DoublePendulumBehavior that acts as a regular double pendulum with a pivot point free to move in the horizontal direction,
# and an optional forcing potential
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
# See https://en.wikipedia.org/wiki/Double_pendulum for more information
#
class GeneralDoublePendulumBehavior(BaseDoublePendulumBehavior):
    
    # Instantiates a GeneralDoublePendulumBehavior with the provided (optional) forcing potential)
    def __init__(self, forcing_potential: Potential = ZeroPotential()):
        self.__forcing_potential = forcing_potential

    # There is a set of 3 linear equations (3 knowns, 3 unknowns) that relate the following six quantities:
    #
    #   theta1_dot
    #   theta2_dot
    #   q_dot
    #   p_theta1
    #   p_theta2
    #   p_q
    #
    # The coefficients for these equations are functions of theta1, theta2, L and d and are returned by this method
    # in the form of a 3x3 matrix, that satsifies the following equation
    #  _          _   _            _         _          _
    # |  A1 A2 A3  | |  theta1_dot  |   =   |  p_theta1  |
    # |  B1 B2 B3  | |  theta2_dot  |       |  p_theta2  |
    # |_ C1 C2 C3 _| |_ q_dot      _|       |_ p_q      _|
    #
    def _coord_dot_p_coord_matrix(self, theta1: float, theta2: float, L: float, m: float, d: float) -> List[List[float]]:
        return np.array([
            [ L**2*5/2 + 2*d**2         , L**2*cos(theta1 - theta2) , 3*L*cos(theta1) ],
            [ L**2*cos(theta1 - theta2) , L**2*1/2 + 2*d**2         , L*cos(theta2)   ],
            [ 3*L*cos(theta1)           , L*cos(theta2)             , 4               ]
        ]) * 1/2*m

    def _p_coord_coord_dot_matrix(self, theta1: float, theta2: float, L: float, m: float, d: float) -> List[List[float]]:
        return np.linalg.inv(self._coord_dot_p_coord_matrix(theta1, theta2, L, m, d))

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
    def _coord_dot_to_p_coord(self, coord_dot: List[float], theta1: float, theta2: float, L: float, m: float, d: float) -> List[float]:
        matrix = self._coord_dot_p_coord_matrix(theta1, theta2, L, m, d)
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
    def _p_coord_to_coord_dot(self, p_coord: List[float], theta1: float, theta2: float, L: float, m: float, d: float) -> List[float]:
        matrix = self._p_coord_coord_dot_matrix(theta1, theta2, L, m, d)
        return np.matmul(matrix, p_coord)

    # Calculates p_theta1_dot given the parameters of the current state and the potential
    def _p_theta1_dot(self, theta1: float, theta2: float, theta1_dot: float, theta2_dot: float, q_dot: float, potential_term: float, L: float, m: float) -> float:
        return 1/2*m * (-3*q_dot*L*theta1_dot*sin(theta1) - L**2*theta1_dot*theta2_dot*sin(theta1 - theta2)) - potential_term

    # Calculates p_theta2_dot given the parameters of the current state and the potential
    def _p_theta2_dot(self, theta1: float, theta2: float, theta1_dot: float, theta2_dot: float, q_dot: float, potential_term: float, L: float, m: float) -> float:
        return 1/2*m * (-1*q_dot*L*theta2_dot*sin(theta2) + L**2*theta1_dot*theta2_dot*sin(theta1 - theta2)) - potential_term

    def state_to_y(self, t: float, state: DoublePendulum.State, prop: DoublePendulum.Properties) -> List[float]:
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

        p_coord = self._coord_dot_to_p_coord([theta1_dot, theta2_dot, q_dot], theta1, theta2, L, m, d)

        return [theta1, theta2, q, p_coord[0], p_coord[1], p_coord[2]]
    
    def y_to_state(self, t: float, y: List[float], prop: DoublePendulum.Properties) -> DoublePendulum.State:
        L = prop.L()
        m = prop.m()
        d = prop.d()

        theta1 = y[0]
        theta2 = y[1]
        q      = y[2]
        p_theta1 = y[3]
        p_theta2 = y[4]
        p_q      = y[5]

        coord_dot = self._p_coord_to_coord_dot([p_theta1, p_theta2, p_q], theta1, theta2, L, m, d)

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
        m = prop.m()
        d = prop.d()

        # Local variables for the y vector elements
        theta1 = y[0]
        theta2 = y[1]
        q      = y[2]
        p_theta1 = y[3]
        p_theta2 = y[4]
        p_q      = y[5]

        # Calculate the time derivatives of each coordinate from the generalized momenta
        coord_dot = self._p_coord_to_coord_dot([p_theta1, p_theta2, p_q], theta1, theta2, L, m, d)
        theta1_dot = coord_dot[0]
        theta2_dot = coord_dot[1]
        q_dot      = coord_dot[2]

        # Store everything in a state instance for passing to functions
        state = DoublePendulum.State(theta1, theta2, q, theta1_dot, theta2_dot, q_dot)

        # Take into account potential due to gravity and our forcing potential
        gravity_terms = self.gravity().dU_dcoord(t, state, prop)
        forcing_terms = self.__forcing_potential.dU_dcoord(t, state, prop)

        # Calculate the time derivatives of the generalized momenta
        p_theta1_dot = self._p_theta1_dot(theta1, theta2, theta1_dot, theta2_dot, q_dot, gravity_terms[0] + forcing_terms[0], L, m)
        p_theta2_dot = self._p_theta2_dot(theta1, theta2, theta1_dot, theta2_dot, q_dot, gravity_terms[1] + forcing_terms[1], L, m)
        p_q_dot      = 0 - (gravity_terms[2] + forcing_terms[2])

        return [theta1_dot, theta2_dot, q_dot, p_theta1_dot, p_theta2_dot, p_q_dot]
    
    def energy_potential(self, t: float, state: DoublePendulum.State, prop: DoublePendulum.Properties) -> float:
        return np.sum(self.gravity().U(t, state, prop) + self.__forcing_potential.U(t, state, prop))

class GeneralSinglePendulumBehavior(BaseDoublePendulumBehavior):
    # Instantiates a GeneralSinglePendulumBehavior with the provided (optional) forcing potential)
    # d_converter should be a function that returns the d value (I = m*d**2) for the whole pendulum, given the properties corresponding to a single arm
    def __init__(self, forcing_potential: Potential = ZeroPotential(), d_converter: Callable[[DoublePendulum.Properties], float] = lambda prop: prop.d()):
        self.__forcing_potential = forcing_potential
        self.__d_converter = d_converter

    # There is a set of 2 linear equations (2 knowns, 2 unknowns) that relate the following four quantities:
    #
    #   theta_dot
    #   q_dot
    #   p_theta
    #   p_q
    #
    # The coefficients for these equations are functions of theta, L and d and are returned by this method
    # in the form of a 2x2 matrix, that satsifies the following equation
    #  _       _   _           _         _         _
    # |  A1 A2  | |  theta_dot  |   =   |  p_theta  |
    # |_ B1 B2 _| |_ q_dot     _|       |_ p_q     _|
    #
    def _coord_dot_p_coord_matrix(self, theta: float, L: float, m: float, d: float) -> List[List[float]]:
        return np.array([
            [ L**2/2 + 2*d**2 , L*cos(theta) ],
            [ L*cos(theta)    , 2            ]
        ]) * 1/2*m

    def _p_coord_coord_dot_matrix(self, theta: float, L: float, m: float, d: float) -> List[List[float]]:
        return np.linalg.inv(self._coord_dot_p_coord_matrix(theta, L, m, d))

    # Transforms a vector of
    #    _           _ 
    #   |  theta_dot  |
    #   |_ q_dot     _|
    #
    # to a vector
    #    _         _
    #   |  p_theta  |
    #   |_ p_q     _|
    #
    # using theta1, theta2, L and d
    #
    def _coord_dot_to_p_coord(self, coord_dot: List[float], theta: float, L: float, m: float, d: float) -> List[float]:
        matrix = self._coord_dot_p_coord_matrix(theta, L, m, d)
        return np.matmul(matrix, coord_dot)

    # Transforms a vector of
    #    _         _ 
    #   |  p_theta  |
    #   |_ p_q     _|
    #
    # to a vector
    #    _           _
    #   |  theta_dot  |
    #   |_ q_dot     _|
    #
    # using theta1, theta2, L and d
    #
    def _p_coord_to_coord_dot(self, p_coord: List[float], theta: float, L: float, m: float, d: float) -> List[float]:
        matrix = self._p_coord_coord_dot_matrix(theta, L, m, d)
        return np.matmul(matrix, p_coord)

    # Calculates p_theta_dot given the parameters of the current state and the potential
    def _p_theta_dot(self, theta: float, theta_dot: float, q_dot: float, potential_term: float, L: float, m: float) -> float:
        return 1/2*m * (-1*q_dot*L*theta_dot*sin(theta)) - potential_term

    # Calculates the net gravitational effect for the whole pendulum
    # Can be applied to the actual potential or the derivative of the potential that is used in the generalized force
    def net_gravity(self, gravity_terms: List[float]) -> float: 
        return (gravity_terms[0] + gravity_terms[1]) / 2

    def state_to_y(self, t: float, state: DoublePendulum.State, prop: DoublePendulum.Properties) -> List[float]:
        # Construct y vector
        L = prop.L() * 2
        m = prop.m() * 2
        d = self.__d_converter(prop)

        # assert(state.theta1() == state.theta2()) TODO check for equivalent angles
        assert(state.theta1_dot() == state.theta2_dot())

        theta     = state.theta1()
        q         = state.q()
        theta_dot = state.theta1_dot()
        q_dot     = state.q_dot()

        p_coord = self._coord_dot_to_p_coord([theta_dot, q_dot], theta, L, m, d)

        return [theta, q, p_coord[0], p_coord[1]]
    
    def y_to_state(self, t: float, y: List[float], prop: DoublePendulum.Properties) -> DoublePendulum.State:
        L = prop.L() * 2
        m = prop.m() * 2
        d = self.__d_converter(prop)

        theta   = y[0]
        q       = y[1]
        p_theta = y[2]
        p_q     = y[3]

        coord_dot = self._p_coord_to_coord_dot([p_theta, p_q], theta, L, m, d)

        return DoublePendulum.State(
            theta1     = theta,
            theta2     = theta,
            q          = q,
            theta1_dot = coord_dot[0],
            theta2_dot = coord_dot[0],
            q_dot      = coord_dot[1]
        )
    
    def dy_dt(self, t: float, y: List[float], prop: DoublePendulum.Properties) -> List[float]:
        L = prop.L() * 2
        m = prop.m() * 2
        d = self.__d_converter(prop)

        # Local variables for the y vector elements
        theta   = y[0]
        q       = y[1]
        p_theta = y[2]
        p_q     = y[3]

        # Calculate the time derivatives of each coordinate from the generalized momenta
        coord_dot = self._p_coord_to_coord_dot([p_theta, p_q], theta, L, m, d)
        theta_dot = coord_dot[0]
        q_dot     = coord_dot[1]

        # Store everything in a state instance for passing to functions
        state = DoublePendulum.State(theta, theta, q, theta_dot, theta_dot, q_dot)

        # Take into account potential due to gravity and our forcing potential
        gravity_terms = self.gravity().dU_dcoord(t, state, prop)
        forcing_terms = self.__forcing_potential.dU_dcoord(t, state, prop)

        gravity_theta = self.net_gravity(gravity_terms)

        assert(forcing_terms[0] == forcing_terms[1]) # TODO is this necessarily true

        # Calculate the time derivatives of the generalized momenta
        p_theta_dot = self._p_theta_dot(theta, theta_dot, q_dot, gravity_theta + forcing_terms[0], L, m)
        p_q_dot      = 0 - (gravity_terms[2] + forcing_terms[2])

        return [theta_dot, q_dot, p_theta_dot, p_q_dot]
    
    def energy_potential(self, t: float, state: DoublePendulum.State, prop: DoublePendulum.Properties) -> float:
        gravity_terms = self.gravity().U(t, state, prop)
        forcing_terms = self.__forcing_potential.U(t, state, prop)
        return self.net_gravity(gravity_terms) + gravity_terms[2] + np.sum(forcing_terms)
    

###################################################################################################################################################################################
# FORCED BEHAVIOR IMPLEMENTATIONS
###################################################################################################################################################################################

# Abstract Base Class for a forcing function on the coordinate q
#
# A forcing function is fully defined by its value as a function of time
# and its derivative as a function of time
class QForcingFunction(ABC):

    @abstractmethod
    def q(self, t: float) -> float:
        pass
    
    @abstractmethod
    def dq_dt(self, t: float) -> float:
        pass

# A forcing function that holds q fixed at a given point (by default, 0)
class FixedQForcingFunction(QForcingFunction):
    
    def __init__(self, fixed_q: float = 0):
        self.__fixed_q = fixed_q

    def q(self, t: float) -> float: return self.__fixed_q
    def dq_dt(self, t: float) -> float: return 0

import sympy as sym
from sympy import symbols, lambdify, diff
sym.init_printing()

# Base class for forcing functions defined by a symbolic algebraic function
class SymbolicForcing(QForcingFunction):

    def __init__(self, q_sym):
        t = symbols('t')
        self.__q_sym = lambdify(t, q_sym)
        self.__dq_dt_sym = lambdify(t, diff(q_sym, t))
    
    def q(self, t: float) -> float: return self.__q_sym(t)
    def dq_dt(self, t: float) -> float: return self.__dq_dt_sym(t)

# Symbolic damped sinusoidal forcing function
#
# A   = amplitude
# w   = angular frequency
# phi = phase
# k   = damping constant
#
def SymbolicSinusoidalForcing(A = 1, w = 1, phi = 0, k = 0):
    t = symbols('t')
    expr = A * sym.exp(-1*k * t) * sym.sin(w*t - phi)
    return expr

# Implementation of DoublePendulumBehavior that forces q and q_dot according to a given function
# instead of solving for them.
#
# This class extends GeneralDoublePendulumBehavior, but overrides the core behavior methods to force q and q_dot.
#
# It also supports a forcing_potential in addition to the forcing function on q
#
class ForcedQDoublePendulumBehavior(GeneralDoublePendulumBehavior):
    
    def __init__(self, forcing_function: QForcingFunction = FixedQForcingFunction(), forcing_potential: Potential = ZeroPotential()):
        super().__init__(forcing_potential = forcing_potential)
        self.__ff = forcing_function
        self.__forcing_potential = forcing_potential

    # Calculates p_theta1 and p_theta2, given theta1, theta2, q_dot and pendulum properties
    #
    # This function uses the matrix coefficients provided by the super class, but re-organizes them to use a given q_dot instead of solving for it
    #
    # The new system of equations can be represented by the following matrix equation (there is no longer an equation for p_q)
    #  _          _   _            _         _          _
    # |  A1 A2 A3  | |  theta1_dot  |   =   |  p_theta1  |
    # |_ B1 B2 B3 _| |  theta2_dot  |       |_ p_theta2 _|
    #                |_ q_dot      _|
    #
    # To invert this system and solve for theta1_dot and theta2_dot we have the following matrix equation:
    #  _     _   _            _         _          _         _          _
    # | A1 A2 | |  theta1_dot  |   =   |  p_theta1  |   -   |  A3*q_dot  |
    # |_B1 B2_| |_ theta2_dot _|       |_ p_theta2 _|       |_ B3*q_dot _|
    #
    # Therefore theta1_dot and theta2_dot can be obtained by multiplying the RHS by the inverse of the 2x2 matrix
    #
    def _p_theta_theta_dot(self, p_theta: List[float], theta1: float, theta2: float, q_dot: float, L: float, m: float, d: float) -> List[float]:
        # Get the original matrix from the superclass
        matrix = self._coord_dot_p_coord_matrix(theta1, theta2, L, m, d)

        # Calculate the RHS of the matrix equation using the top-right and middle-right coefficients from the matrix
        p_theta_adj = p_theta - q_dot * matrix[0:2, 2]

        # Extract the 2x2 top-left sub-matrix (i.e. the coefficients multiplying only theta1_dot and theta2_dot)
        matrix_adj = matrix[0:2, 0:2]

        # Invert the matrix and multiply by the RHS to solve for theta1_dot and theta2_dot
        theta_dot = np.matmul(np.linalg.inv(matrix_adj), p_theta_adj)
        return theta_dot

    def state_to_y(self, t: float, state: DoublePendulum.State, prop: DoublePendulum.Properties) -> List[float]:
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

        # Enforce that the state of q and q_dot always matches the forcing function values
        assert(q == self.__ff.q(t))
        assert(q_dot == self.__ff.dq_dt(t))

        p_coord = self._coord_dot_to_p_coord([theta1_dot, theta2_dot, q_dot], theta1, theta2, L, m, d)

        # The y-vector representation no longer has q and p_q_dot since they are not part of the differential equation when they are forced
        return [theta1, theta2, p_coord[0], p_coord[1]]
    
    def y_to_state(self, t: float, y: List[float], prop: DoublePendulum.Properties) -> DoublePendulum.State:
        L = prop.L()
        m = prop.m()
        d = prop.d()

        theta1 = y[0]
        theta2 = y[1]
        p_theta1 = y[2]
        p_theta2 = y[3]

        # Get q and q_dot from the forcing function
        q = self.__ff.q(t)
        q_dot = self.__ff.dq_dt(t)

        # Calculate theta1_dot and theta2_dot from p_theta1_dot, p_theta2_dot and q_dot
        theta_dot = self._p_theta_theta_dot([p_theta1, p_theta2], theta1, theta2, q_dot, L, m, d)
        
        return DoublePendulum.State(
            theta1     = theta1,
            theta2     = theta2,
            q          = q,
            theta1_dot = theta_dot[0],
            theta2_dot = theta_dot[1],
            q_dot      = q_dot
        )
        
    def dy_dt(self, t: float, y: List[float], prop: DoublePendulum.Properties) -> List[float]:
        L = prop.L()
        m = prop.m()
        d = prop.d()

        theta1 = y[0]
        theta2 = y[1]
        p_theta1 = y[2]
        p_theta2 = y[3]

        # Get q and q_dot from the forcing function
        q = self.__ff.q(t)
        q_dot = self.__ff.dq_dt(t)

        # Calculate theta1_dot and theta2_dot from p_theta1_dot, p_theta2_dot and q_dot
        theta_dot = self._p_theta_theta_dot([p_theta1, p_theta2], theta1, theta2, q_dot, L, m, d)
        theta1_dot = theta_dot[0]
        theta2_dot = theta_dot[1]

        # Store everything in a state instance for passing to functions
        state = DoublePendulum.State(theta1, theta2, q, theta1_dot, theta2_dot, q_dot)

        # Take into account potential due to gravity and our forcing potential
        gravity_terms = self.gravity().dU_dcoord(t, state, prop)
        forcing_terms = self.__forcing_potential.dU_dcoord(t, state, prop)

        # Calculate the time derivatives of the generalized momenta
        p_theta1_dot = self._p_theta1_dot(theta1, theta2, theta1_dot, theta2_dot, q_dot, gravity_terms[0] + forcing_terms[0], L, m)
        p_theta2_dot = self._p_theta2_dot(theta1, theta2, theta1_dot, theta2_dot, q_dot, gravity_terms[1] + forcing_terms[1], L, m)

        return [theta1_dot, theta2_dot, p_theta1_dot, p_theta2_dot]

# Implementation of DoublePendulumBehavior that behaves as a single pendulum while forcing q and q_dot according to a given function
# instead of solving for them.
#
# This class extends GeneralSinglePendulumBehavior, but overrides the core behavior methods to force q and q_dot.
#
# It also supports a forcing_potential in addition to the forcing function on q
#
class ForcedQSinglePendulumBehavior(GeneralSinglePendulumBehavior):
    
    def __init__(self, forcing_function: QForcingFunction = FixedQForcingFunction(), forcing_potential: Potential = ZeroPotential(), d_converter: Callable[[DoublePendulum.Properties], float] = lambda prop: prop.d()):
        super().__init__(forcing_potential = forcing_potential, d_converter = d_converter)
        self.__ff = forcing_function
        self.__forcing_potential = forcing_potential
        self.__d_converter = d_converter

    def _theta_dot_to_p_theta(self, t: float, theta_dot: float, theta: float, L: float, m: float, d: float) -> float:
        # Get q_dot from the forcing function
        q_dot = self.__ff.dq_dt(t)

        return 1/2*m * ((L**2/2 + 2*d**2)*theta_dot + L*cos(theta)*q_dot)

    def _p_theta_to_theta_dot(self, t: float, p_theta: float, theta: float, L: float, m: float, d: float) -> float:
        # Get q_dot from the forcing function
        q_dot = self.__ff.dq_dt(t)

        return (p_theta / (1/2*m) - L*cos(theta)*q_dot) / (L**2/2 + 2*d**2)

    # Calculates p_theta_dot given the parameters of the current state and the potential
    def _p_theta_dot(self, t: float, theta: float, theta_dot: float, potential_term: float, L: float, m: float) -> float:
        # Get q_dot from the forcing function
        q_dot = self.__ff.dq_dt(t)

        return 1/2*m * (-1*q_dot*L*theta_dot*sin(theta)) - potential_term
    
    def state_to_y(self, t: float, state: DoublePendulum.State, prop: DoublePendulum.Properties) -> List[float]:
        # Construct y vector
        L = prop.L() * 2
        m = prop.m() * 2
        d = self.__d_converter(prop)

        # assert(state.theta1() == state.theta2()) TODO check for equivalent angles
        assert(state.theta1_dot() == state.theta2_dot())

        theta     = state.theta1()
        q         = state.q()
        theta_dot = state.theta1_dot()
        q_dot     = state.q_dot()

        # Enforce that the state of q and q_dot always matches the forcing function values
        assert(q == self.__ff.q(t))
        assert(q_dot == self.__ff.dq_dt(t))

        p_theta = self._theta_dot_to_p_theta(t, theta_dot, theta, L, m, d)

        return [theta, p_theta]
    
    def y_to_state(self, t: float, y: List[float], prop: DoublePendulum.Properties) -> DoublePendulum.State:
        L = prop.L() * 2
        m = prop.m() * 2
        d = self.__d_converter(prop)

        theta   = y[0]
        p_theta = y[1]
        
        # Get q and q_dot from the forcing function
        q     = self.__ff.q(t)
        q_dot = self.__ff.dq_dt(t)

        theta_dot = self._p_theta_to_theta_dot(t, p_theta, theta, L, m, d)

        return DoublePendulum.State(
            theta1     = theta,
            theta2     = theta,
            q          = q,
            theta1_dot = theta_dot,
            theta2_dot = theta_dot,
            q_dot      = q_dot
        )
    
    def dy_dt(self, t: float, y: List[float], prop: DoublePendulum.Properties) -> List[float]:
        L = prop.L() * 2
        m = prop.m() * 2
        d = self.__d_converter(prop)

        # Local variables for the y vector elements
        theta   = y[0]
        p_theta = y[1]

        # Get q and q_dot from the forcing function
        q     = self.__ff.q(t)
        q_dot = self.__ff.dq_dt(t)

        # Calculate the time derivatives of each coordinate from the generalized momenta
        theta_dot = self._p_theta_to_theta_dot(t, p_theta, theta, L, m, d)

        # Store everything in a state instance for passing to functions
        state = DoublePendulum.State(theta, theta, q, theta_dot, theta_dot, q_dot)

        # Take into account potential due to gravity and our forcing potential
        gravity_terms = self.gravity().dU_dcoord(t, state, prop)
        forcing_terms = self.__forcing_potential.dU_dcoord(t, state, prop)

        gravity_theta = self.net_gravity(gravity_terms)

        assert(forcing_terms[0] == forcing_terms[1]) # TODO is this necessarily true

        # Calculate the time derivatives of the generalized momenta
        p_theta_dot = self._p_theta_dot(t, theta, theta_dot, gravity_theta + forcing_terms[0], L, m)

        return [theta_dot, p_theta_dot]