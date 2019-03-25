from typing import List, Tuple, Dict

import sympy as sp
import scipy.constants

from math import pi

sp.init_printing()

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
            `I`      : moment of inertia of the pendulum about an axis passing through its center,
                       perpendicular to the plane of oscillation
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
            `x`     : x coordinate (as a symbolic function of time) of the starting endpoint of the pendulum
            `y`     : y coordinate (as a symbolic function of time) of the starting endpoint of the pendulum
            `theta` : angle of the pendulum (as a symbolic function of time) with respect to the vertical through (x,y)
        """

        def __init__(self, x: sp.Function, y: sp.Function, theta: sp.Function):
            self._x = x
            self._y = y
            self._theta = theta
        
        @property
        def x(self)     -> sp.Function : return self._x
        @property
        def y(self)     -> sp.Function : return self._y
        @property
        def theta(self) -> sp.Function : return self._theta

    class State:
        """Holds the numeric values required to define the current instantaneous motion of the pendulum

        Attributes:
            `x`     : x coordinate of the starting endpoint of the pendulum
            `x_dot` : x velocity of the starting endpoint of the pendulum
            `y`     : y coordinate of the starting endpoint of the pendulum
            `y_dot` : y velocity of the starting endpoint of the pendulum
            `theta`     : angle of the pendulum with respect to the vertical through (x,y)
            `theta_dot` : anglular velocity of the pendulum about an axis on the pendulum,
                          perpendicular to the plane of oscillation
        """

        def __init__(self, x: float, y: float, theta: float, x_dot: float, y_dot: float, theta_dot: float):
            self._x         = x
            self._y         = y
            self._theta     = theta
            self._x_dot     = x_dot
            self._y_dot     = y_dot
            self._theta_dot = theta_dot
        
        @property
        def x(self)         -> float : return self._x
        @property
        def y(self)         -> float : return self._y
        @property
        def theta(self)     -> float : return self._theta
        @property
        def x_dot(self)     -> float : return self._x_dot
        @property
        def y_dot(self)     -> float : return self._y_dot
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

    def start(self, t: sp.Symbol) -> Tuple[sp.Expr, sp.Expr]:
        """Generates symbolic expressions for the coordinates of the start endpoint of the pendulum
        
        Arguments:
            `t` : a symbol for the time variable

        Returns: Tuple of (x, y) where each coordinate is a symbolic expression
        """
        return (self.sym.x, self.sym.y)

    def com(self, t: sp.Symbol) -> Tuple[sp.Expr, sp.Expr]:
        """Generates symbolic expressions for the coordinates of the centre of mass of the pendulum

        Arguments:
            `t` : a symbol for the time variable

        Returns:
            Tuple of (x_COM, y_COM) where each coordinate is a symbolic expression
        """
        L = self.prop.L
        x = self.sym.x
        y = self.sym.y
        theta = self.sym.theta

        x_com = x(t) + L/2 * sp.sin(theta(t))
        y_com = y(t) + L/2 * sp.cos(theta(t))

        return (x_com, y_com)

    def end(self, t: sp.Symbol) -> Tuple[sp.Expr, sp.Expr]:
        """Generates symbolic expressions for the coordinates of the end endpoint of the pendulum

        Arguments:
            `t` : a symbol for the time variable

        Returns:
            Tuple of (x_end, y_end) where each coordinate is a symbolic expression
        """
        L = self.prop.L
        x = self.sym.x
        y = self.sym.y
        theta = self.sym.theta

        x_end = x(t) + L * sp.sin(theta(t))
        y_end = y(t) + L * sp.cos(theta(t))

        return (x_end, y_end)

    def U(self, t: sp.Symbol) -> sp.Expr:
        """Generates a symbolic expression for the potential energy of the pendulum

        The zero of potential energy is taken to be at a y coordinate of 0

        Arguments:
            `t` : a symbol for the time variable

        Returns:
            A symbolic expression for the potential energy of the pendulum
        """
        m = self.prop.m
        _, y_com = self.com(t)

        g = scipy.constants.g

        return m * g * y_com

    def T(self, t: sp.Symbol) -> sp.Expr:
        """Generates and returns a symbolic expression for the kinetic energy of the pendulum

        This takes into account both the translation and rotational kinetic energies

        Arguments:
            `t` : a symbol for the time variable

        Returns:
            A symbolic expression for the kinetic energy of the pendulum
        """
        m = self.prop.m
        I = self.prop.I
        theta = self.sym.theta(t)
        x_com, y_com = self.com(t)

        x_dot = sp.diff(x_com, t)
        y_dot = sp.diff(y_com, t)
        theta_dot = sp.diff(theta, t)

        T_translation = 1/2*m * (x_dot**2 + y_dot**2)
        T_rotation = 1/2*I * theta_dot**2

        return T_translation + T_rotation