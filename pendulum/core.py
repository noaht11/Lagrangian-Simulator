from typing import List, Tuple, Dict
from abc import ABC, abstractmethod

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
# PENDULUM DEFINITIONS
###################################################################################################################################################################################

class Pendulum(ABC):
    """Abstract base class for representing a pendulum

    A pendulum is considered to be any body (could be a point mass or a rigid body) suspended from a pivot that will oscillate about an equilibrium position due to gravity.

    The following properties are considered to be common to all pendulums and are therefore declared/implemented in this base class:

        1) `Coordinates`

            The Coordiantes are symbolic functions that represent the possible degrees of freedom of the pendulum.
            These are stored symbolically (as SymPy Functions) to allow the derivation of the Lagrangian equations.

        2) `State`:
            
            The State is a set of numeric values for the current values of the coordinates and their first time derivatives.
            These values are to be used when numerically solving the differential equations of motion of the pendulum.

    The Pendulum class is MUTABLE, since the State can be updated through the `set_state` method.
    This is so that the pendulum can easily be evolved through time, by anyone capable of solving the appropriate Lagrange equations.

    Note:
        Although the Pendulum itself is mutable, the State class is immutable (the Pendulum is modified by providing a new instance of the State class).
        This means instances of the State class can safely be used to pass information about the state of the pendulum without risk of it being modified.
    
    This class further defines the following abstract methods as they are considered to exist for all pendulums, but their implementation depends on the type of pendulum:

        - `end` : returns the location of the endpoint
        - `COM` : returns the location of the center of mass
        - `U`   : returns the potential energy
        - `T`   : returns the kinetic energy

    See the documentation of these methods for more information.
    
    """
    
    class Coordinates:
        """Holds the symbolic functions that represent the possible degrees of freedom of the pendulum

        This class is IMMUTABLE.

        Attributes:
            `x`     : x coordinate (as a symbolic function of time) of the pivot of the pendulum
            `y`     : y coordinate (as a symbolic function of time) of the pivot of the pendulum
            `theta` : angle of the pendulum (as a symbolic function of time) with respect to the vertical through the pivot
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

        This class is IMMUTABLE.

        Attributes:
            `x`     : x coordinate of the pivot of the pendulum
            `x_dot` : x velocity of the pivot of the pendulum
            `y`     : y coordinate of the pivot of the pendulum
            `y_dot` : y velocity of the pivot of the pendulum
            `theta`     : angle of the pendulum with respect to the vertical through the pivot
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

    def __init__(self, coordinates: Coordinates, state: State):
        self._coordinates = coordinates
        self._state = state

    @property
    def coordinates(self) -> Coordinates: return self._coordinates
    @property
    def state(self)       -> State:       return self._state

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

    def pivot(self, t: sp.Symbol) -> Tuple[sp.Expr, sp.Expr]:
        """Generates symbolic expressions for the coordinates of the pivot of the pendulum
        
        Arguments:
            `t` : a symbol for the time variable

        Returns: Tuple of (x, y) where each coordinate is a symbolic expression
        """
        return (self.sym.x(t), self.sym.y(t))
    
    @abstractmethod
    def end(self, t: sp.Symbol) -> Tuple[sp.Expr, sp.Expr]:
        """Generates symbolic expressions for the coordinates of the endpoint of the pendulum

        The endpoint of the pendulum is the point where a subsequent pendulum would be attached if a chain of n pendulums were to be constructed.

        Arguments:
            `t` : a symbol for the time variable

        Returns:
            Tuple of (x_end, y_end) where each coordinate is a symbolic expression
        """
        pass

    @abstractmethod
    def COM(self, t: sp.Symbol) -> Tuple[sp.Expr, sp.Expr]:
        """Generates symbolic expressions for the coordinates of the centre of mass (COM) of the pendulum

        Arguments:
            `t` : a symbol for the time variable

        Returns:
            Tuple of (x_COM, y_COM) where each coordinate is a symbolic expression
        """
        pass

    @abstractmethod
    def U(self, t: sp.Symbol) -> sp.Expr:
        """Generates a symbolic expression for the potential energy of the pendulum

        The zero of potential energy is taken to be at a y coordinate of 0

        Arguments:
            `t` : a symbol for the time variable

        Returns:
            A symbolic expression for the potential energy of the pendulum
        """
        pass
    
    @abstractmethod
    def T(self, t: sp.Symbol) -> sp.Expr:
        """Generates and returns a symbolic expression for the kinetic energy of the pendulum

        Arguments:
            `t` : a symbol for the time variable

        Returns:
            A symbolic expression for the kinetic energy of the pendulum
        """
        pass

class CompoundPendulum(Pendulum):
    """Implementation of the Pendulum class for a single compound pendulum (a pendulum whose mass is evenly distributed along its length)

    Attributes:
        `prop`  : physical attributes of the pendulum that do not change over time
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

    def __init__(self, coordiantes: Pendulum.Coordinates, state: Pendulum.State, prop: Properties):
        super().__init__(coordiantes, state)
        self._prop = prop
    
    @property
    def prop(self) -> Properties: return self._prop

    def end(self, t: sp.Symbol) -> Tuple[sp.Expr, sp.Expr]:
        """Generates symbolic expressions for the coordinates of the endpoint of the pendulum

        Arguments:
            `t` : a symbol for the time variable

        Returns:
            Tuple of (x_end, y_end) where each coordinate is a symbolic expression
        """
        L = self.prop.L
        x = self.coordinates.x
        y = self.coordinates.y
        theta = self.coordinates.theta

        x_end = x(t) + L * sp.sin(theta(t))
        y_end = y(t) + L * sp.cos(theta(t))

        return (x_end, y_end)

    def COM(self, t: sp.Symbol) -> Tuple[sp.Expr, sp.Expr]:
        L = self.prop.L
        x = self.coordinates.x
        y = self.coordinates.y
        theta = self.coordinates.theta

        x_com = x(t) + L/2 * sp.sin(theta(t))
        y_com = y(t) + L/2 * sp.cos(theta(t))

        return (x_com, y_com)

    def U(self, t: sp.Symbol) -> sp.Expr:
        m = self.prop.m
        _, y_COM = self.COM(t)

        g = scipy.constants.g

        return m * g * y_COM

    def T(self, t: sp.Symbol) -> sp.Expr:
        m = self.prop.m
        I = self.prop.I
        theta = self.coordinates.theta(t)
        x_COM, y_COM = self.COM(t)

        x_dot = sp.diff(x_COM, t)
        y_dot = sp.diff(y_COM, t)
        theta_dot = sp.diff(theta, t)

        T_translation = 1/2*m * (x_dot**2 + y_dot**2)
        T_rotation = 1/2*I * theta_dot**2

        return T_translation + T_rotation