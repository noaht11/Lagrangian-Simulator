from typing import List, Tuple, Dict
from abc import ABC, abstractmethod
from math import pi

import sympy as sp
import scipy.constants

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

class Physics(ABC):
    """Abstract base class for representing the physical properties and behavior of the pendulum
    
    This class defines the following abstract methods as they are considered to exist for all pendulums, but their implementation depends on the type of pendulum:

        - `endpoint` : returns the location of the endpoint
        - `COM`      : returns the location of the center of mass
        - `U`        : returns the potential energy
        - `T`        : returns the kinetic energy

    See the documentation of these methods for more information.
    """

    @abstractmethod
    def endpoint(self, t: sp.Symbol, coordinates: Coordinates) -> Tuple[sp.Expr, sp.Expr]:
        """Generates symbolic expressions for the coordinates of the endpoint of the pendulum

        The endpoint of the pendulum is the point where a subsequent pendulum would be attached if a chain of pendulums were to be constructed.

        Arguments:
            `t` : a symbol for the time variable

        Returns:
            Tuple of (x_end, y_end) where each coordinate is a symbolic expression
        """
        pass

    @abstractmethod
    def COM(self, t: sp.Symbol, coordinates: Coordinates) -> Tuple[sp.Expr, sp.Expr]:
        """Generates symbolic expressions for the coordinates of the centre of mass (COM) of the pendulum

        Arguments:
            `t` : a symbol for the time variable

        Returns:
            Tuple of (x_COM, y_COM) where each coordinate is a symbolic expression
        """
        pass

    @abstractmethod
    def U(self, t: sp.Symbol, coordinates: Coordinates) -> sp.Expr:
        """Generates a symbolic expression for the potential energy of the pendulum

        The zero of potential energy is taken to be at a y coordinate of 0

        Arguments:
            `t` : a symbol for the time variable

        Returns:
            A symbolic expression for the potential energy of the pendulum
        """
        pass
    
    @abstractmethod
    def T(self, t: sp.Symbol, coordinates: Coordinates) -> sp.Expr:
        """Generates and returns a symbolic expression for the kinetic energy of the pendulum

        Arguments:
            `t` : a symbol for the time variable

        Returns:
            A symbolic expression for the kinetic energy of the pendulum
        """
        pass

class Pendulum():
    """General class for representing a pendulum

    A pendulum is considered to be any body (could be a point mass or a rigid body) suspended from a pivot that will oscillate about an equilibrium position due to gravity.

    The following properties are considered to be common to all pendulums and are therefore declared/implemented in this base class:

        1) `Coordinates`

            The Coordinates are symbolic functions that represent the possible degrees of freedom of the pendulum.
            These are stored symbolically (as SymPy Functions) to allow the derivation of the Lagrangian equations.

        2) `State`:
            
            The State is a set of numeric values for the current values of the coordinates and their first time derivatives.
            These values are to be used when numerically solving the differential equations of motion of the pendulum.
        
        3) `Physics`:

            The Physics is an object that represents the physical properties and behavior of the pendulum
            `Pendulum.Physics` is an abstract base class, so specific types of pendulums should subclass it to define their properties and behavior

    The Pendulum class is MUTABLE, since the State can be updated through the `set_state` method.
    This is so that the pendulum can easily be evolved through time, by anyone capable of solving the appropriate Lagrange equations.

    Note:
        Although the Pendulum itself is mutable, the State class is immutable (the Pendulum is modified by providing a new instance of the State class).
        This means instances of the State class can safely be used to pass information about the state of the pendulum without risk of it being modified.
    """
    
    def __init__(self, coordinates: Coordinates, state: State, physics: Physics):
        self._coordinates = coordinates
        self._state = state
        self._physics = physics

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
        adj_state = State(
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
    
    def endpoint(self, t: sp.Symbol) -> Tuple[sp.Expr, sp.Expr]:
        """Wrapper for `Physics.endpoint`"""
        return self._physics.endpoint(t, self.coordinates)

    def COM(self, t: sp.Symbol) -> Tuple[sp.Expr, sp.Expr]:
        """Wrapper for `Physics.COM`"""
        return self._physics.COM(t, self.coordinates)

    def U(self, t: sp.Symbol) -> sp.Expr:
        """Wrapper for `Physics.U`"""
        return self._physics.U(t, self.coordinates)
    
    def T(self, t: sp.Symbol) -> sp.Expr:
        """Wrapper for `Physics.T`"""
        return self._physics.T(t, self.coordinates)