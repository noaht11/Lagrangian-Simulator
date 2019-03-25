from abc import ABC, abstractmethod
from typing import List, Tuple, Callable, Dict

import sympy as sp

from math import sin, cos, pi, inf, exp

###################################################################################################################################################################################
# UTILITY FUNCTIONS
###################################################################################################################################################################################

def neg_pi_to_pi(theta: float) -> float:
    """Converts an angle in radians to the equivalent angle in radians constrained between -pi and pi, where 0 remains at angle 0
    """
    modded = theta % (2*pi)
    return modded + (modded > pi) * (-2*pi)

###################################################################################################################################################################################
# PENDULUM DEFINITION
###################################################################################################################################################################################

class Pendulum:
    """Represents the properties, math, and state of a single pendulum

    The state of the pendulum is represented numerically while the math governing the energies of the pendulum
    is represented symbolically using SymPy expressions.
    This allows the Lagrangian equations of motion to be simplified symbolically before being integrated numerically.

    Attributes:
        `prop`  : attributes of the pendulum that do not change over time
        `sym`   : symbolic quantities required to define the Lagrangian equations for the pendulum
        `state` : numeric values required to define the current instantaneous motion of the pendulum
    """
    
    class Properties:
        """Represents all attributes of the pendulum that do not change over time:

        Attributes:
            `L`      : length of the pendulum
            `m`      : total mass of the pendulum
            `I`      : moment of inertia of the pendulum about an axis passing through its center, perpendicular to the plane of oscillation
            `extras` : additional, instance-specific properties
        """

        def __init__(self, L: float, m: float, I: float, **extras):
            self._L = L
            self._m = m
            self._I = I
            self._extras = extras
        
        @property
        def L(self) -> float: return self._L
        @property
        def m(self) -> float: return self._m
        @property
        def I(self) -> float: return self._I
        @property
        def extras(self) -> Dict[str, float]: return self._extras

    class Symbols:
        """Represents the symbolic quantities required to define the Lagrangian equations for the pendulum:

        Attributes:
            `x`     : x coordinate of the endpoint of the pendulum, closest to the first element in a chain of pendulums if applicable
            `y`     : y coordinate of the endpoint of the pendulum, closest to the first element in a chain of pendulums if applicable
            `theta` : angle of the pendulum with respect to the vertical through (x,y)
        """

        def __init__(self, x: sp.Expr, y: sp.Expr, theta: sp.Symbol):
            self._x = x
            self._y = y
            self._theta = theta
        
        @property
        def x(self)     -> sp.Expr   : return self._x
        @property
        def y(self)     -> sp.Expr   : return self._y
        @property
        def theta(self) -> sp.Symbol : return self._theta

    class State:
        """Holds the numeric values required to define the current instantaneous motion of the pendulum

        Attributes:
            `theta`     : angle of the pendulum with respect to the vertical through (x,y)
            `theta_dot` : anglular velocity of the pendulum about an axis on the pendulum, perpendicular to the plane of oscillation
        """

        def __init__(self, theta: float = 0, theta_dot: float = 0):
            self._theta     = theta
            self._theta_dot = theta_dot
        
        @property
        def theta(self)     -> float : return self._theta
        @property
        def theta_dot(self) -> float : return self._theta_dot
        
        def __str__(self) -> str:
                return str([self.theta(), self.theta_dot()])

    def __init__(self, prop: Properties, sym: Symbols, state: State):
        self._prop = prop
        self._sym = sym
        self._state = state
    
    @property
    def prop(self)  -> Properties: return self._prop
    @property
    def sym(self)   -> Symbols:    return self._sym
    @property
    def state(self) -> State:      return self._state

    def set_state(self, new_state: State):
        """Updates the current state of the pendulum
        
        Arguments:
            `new_state` : the new state of the pendulum
        """

        # Constrain the angles to between -pi and pi for easy interpretation
        adj_state = Pendulum.State(
            neg_pi_to_pi(new_state.theta()),
            new_state.theta_dot()
        )
        self._state = adj_state

    def start(self) -> Tuple[sp.Expr, sp.Expr]:
        """Calculates expressions for the coordinates of the start endpoint of the pendulum
        
        Returns: Tuple of (x, y) where each coordinate is a symbolic expression
        """
        return (self.sym.x, self.sym.y)

    def com(self) -> Tuple[sp.Expr, sp.Expr]:
        """Calculates expressions for the coordinates of the centre of mass of the pendulum

        Returns: Tuple of (x_COM, y_COM) where each coordinate is a symbolic expression
        """
        L = self.prop.L
        theta = self.sym.theta

        x_com = self.sym.x + L/2 * sp.sin(theta)
        y_com = self.sym.y + L/2 * sp.cos(theta)

        return (x_com, y_com)

    def end(self) -> Tuple[sp.Expr, sp.Expr]:
        """Calculates expressions for the coordinates of the end endpoint of the pendulum

        Returns: Tuple of (x_end, y_end) where each coordinate is a symbolic expression
        """
        L = self.prop.L
        theta = self.sym.theta

        x_end = self.sym.x + L * sp.sin(theta)
        y_end = self.sym.y + L * sp.cos(theta)

        return (x_end, y_end)

    def energy_potential(self) -> sp.Expr:
        L = self.prop.L
        m = self.prop.m

        return m * L / 2 * sp.cos(self.sym.theta)

    def energy_kinetic(self) -> sp.Expr:
        pass

    # # Calculates and returns the kinetic energy of the current state of the pendulum
    # def energy_kinetic(self) -> float:
    #     # Local copies of the state variables for convenience
    #     L = self.prop().L()
    #     m = self.prop().m()
    #     d = self.prop().d()
    #     theta1 = self.state().theta1()
    #     theta2 = self.state().theta2()
    #     q = self.state().q()
    #     theta1_dot = self.state().theta1_dot()
    #     theta2_dot = self.state().theta2_dot()
    #     q_dot = self.state().q_dot()

    #     # This is just the final formula for kinetic energy after a lot of re-arranging, so there's no good way to decompose it
    #     T = 1/2*m * (2*q_dot**2   +   q_dot*L*(3*cos(theta1)*theta1_dot + cos(theta2)*theta2_dot)   +   L**2*(1/4*(5*theta1_dot**2 + theta2_dot**2) + theta1_dot*theta2_dot*cos(theta1 - theta2))   +   d**2*(theta1_dot**2 + theta2_dot**2))

    #     return T