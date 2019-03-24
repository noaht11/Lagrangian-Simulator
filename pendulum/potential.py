from abc import ABC, abstractmethod
from typing import List, Tuple, Callable

import scipy.constants

from pendulum.core import *

###################################################################################################################################################################################
# ABSTRACT BASE CLASSES
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
    def U(self, t: float, state: DoublePendulum.State, prop: DoublePendulum.Properties) -> List[float]:
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
    def dU_dcoord(self, t: float, state: DoublePendulum.State, prop: DoublePendulum.Properties) -> float:
        pass


###################################################################################################################################################################################
# BASE POTENTIALS
###################################################################################################################################################################################

# A potential that is the sum of two other potentials
class SumPotential(Potential):

    def __init__(self, potentialA: Potential, potentialB: Potential, subtract: bool = False):
        self.__potentialA = potentialA
        self.__potentialB = potentialB
        self.__subtract = subtract
    
    def op(self):
        return (-1 if self.__subtract else 1)

    def U(self, t: float, state: DoublePendulum.State, prop: DoublePendulum.Properties) -> List[float]:
        return np.array(self.__potentialA.U(t, state, prop)) + self.op() * np.array(self.__potentialB.U(t, state, prop))
    
    def dU_dcoord(self, t: float, state: DoublePendulum.State, prop: DoublePendulum.Properties) -> List[float]:
        return np.array(self.__potentialA.dU_dcoord(t, state, prop)) + self.op() * np.array(self.__potentialB.dU_dcoord(t, state, prop))

# A base implementation of Potential that adds support for using the + and - operators
#
# All custom defined potentials should inherit from BasePotential instead of Potential,
# so that they support these operators
#
class BasePotential(Potential):

    def __add__(self, other):
        return SumPotential(self, other, subtract = False)
    
    def __sub__(self, other):
        return SumPotential(self, other, subtract = True)


###################################################################################################################################################################################
# POTENTIAL IMPLEMENTATIONS
###################################################################################################################################################################################

# A potential that is zero everywhere
class ZeroPotential(BasePotential):

    def U(self, t: float, state: DoublePendulum.State, prop: DoublePendulum.Properties) -> List[float]:
        return [0,0,0]
    
    def dU_dcoord(self, t: float, state: DoublePendulum.State, prop: DoublePendulum.Properties) -> List[float]:
        return [0,0,0]

# The potential energy due to gravity where 0 is at the height of the pivot point (q)
class GravitationalPotential(BasePotential):
    # Fundamental constants:
    g = scipy.constants.g

    def U(self, t: float, state: DoublePendulum.State, prop: DoublePendulum.Properties) -> List[float]:
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
    
    def dU_dcoord(self, t: float, state: DoublePendulum.State, prop: DoublePendulum.Properties) -> List[float]:
        g = GravitationalPotential.g

        # Local copies of the state variables for convenience
        L = prop.L()
        m = prop.m()
        theta1 = state.theta1()
        theta2 = state.theta2()

        dU_dtheta1 = -3/2*m*g*L*sin(theta1)
        dU_dtheta2 = -1/2*m*g*L*sin(theta2)
        dU_dq = 0 # q has no effect on gravitational potential

        return [dU_dtheta1, dU_dtheta2, dU_dq]

# A harmonic oscillator potential where:
#
#   U(coord) = 1/2 * coord_k * (coord - coord_eq)^2
#
class HarmonicOscillatorPotential(BasePotential):

    def __init__(self, theta1_k: float = 0, theta1_eq: float = 0, theta2_k: float = 0, theta2_eq: float = 0, q_k: float = 0, q_eq: float = 0):
        self.__theta1_k  = theta1_k
        self.__theta1_eq = theta1_eq
        self.__theta2_k  = theta2_k
        self.__theta2_eq = theta2_eq
        self.__q_k       = q_k
        self.__q_eq      = q_eq

    def U(self, t: float, state: DoublePendulum.State, prop: DoublePendulum.Properties) -> List[float]:
        return [
            1/2 * self.__theta1_k * (state.theta1() - self.__theta1_eq)**2,
            1/2 * self.__theta2_k * (state.theta2() - self.__theta2_eq)**2,
            1/2 * self.__q_k      * (state.q()      - self.__q_eq     )**2
        ]
    
    def dU_dcoord(self, t: float, state: DoublePendulum.State, prop: DoublePendulum.Properties) -> List[float]:
        return [
            self.__theta1_k * (state.theta1() - self.__theta1_eq),
            self.__theta2_k * (state.theta2() - self.__theta2_eq),
            self.__q_k      * (state.q()      - self.__q_eq     )
        ]

FIXED_APPROX_COEFF = 1E2

# A potential that approximately fixes q (the pivot point) in place
#
# This is accomplished by created an extremely strong harmonic oscillator potential
# with a stable equilibrium at q = q_eq such that q would need absurd amounts of
# kinetic energy to leave that equilibrium position
#
def FixedQPotential(q_eq: float = 0):
    return HarmonicOscillatorPotential(q_k = FIXED_APPROX_COEFF, q_eq = q_eq)

# A potential that approximately fixes theta1 = theta2, so that the pendulum behaves like a single pendulum
#
# This is accomplished by created an extremely strong harmonic oscillator potential
# with a stable equilibrium at theta1 = theta2 such that you would need absurd amounts of
# kinetic energy to leave that equilibrium position
#
class SinglePendulumPotential(BasePotential):

    def U(self, t: float, state: DoublePendulum.State, prop: DoublePendulum.Properties) -> List[float]:
        return [
            1/2 * FIXED_APPROX_COEFF * (state.theta1() - state.theta2())**2,
            1/2 * FIXED_APPROX_COEFF * (state.theta2() - state.theta1())**2,
            0
        ]
    
    def dU_dcoord(self, t: float, state: DoublePendulum.State, prop: DoublePendulum.Properties) -> List[float]:
        return [
            FIXED_APPROX_COEFF * (state.theta1() - state.theta2()),
            FIXED_APPROX_COEFF * (state.theta2() - state.theta1()),
            0
        ]
