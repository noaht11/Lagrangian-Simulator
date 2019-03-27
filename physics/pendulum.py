from typing import List, Tuple, Dict
from abc import ABC, abstractmethod
from math import pi

import sympy as sp
import scipy.constants

from pendulum.lagrangian import LagrangianBody, Constraint, unconstrained_DoF

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
# BASE CLASSES
###################################################################################################################################################################################

class SinglePendulum(LagrangianBody):
    """Implementation of a single pendulum as a lagrangian body

    A single pendulum is considered to have 3 degrees of freedom:

        1) x coordinate of the pivot
        2) y coordinate of the pivot
        3) angle of the pendulum

    This class is IMMUTABLE.

    Attributes:
        `x`     : x coordinate (as a symbolic function of time) of the pivot of the pendulum
        `y`     : y coordinate (as a symbolic function of time) of the pivot of the pendulum
        `theta` : angle of the pendulum (as a symbolic function of time) with respect to the vertical through the pivot
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

    def __init__(self, coordinates: Coordinates, physics: Physics):
        self._coordinates = coordinates
        self._physics = physics
    
    @property
    def coordinates(self) -> Coordinates : return self._coordinates

    @property
    def DoF(self) -> List[sp.Function]:
        """Implementation of superclass method"""
        return [self.coordinates.x, self.coordinates.y, self.coordinates.theta]
    
    def endpoint(self, t: sp.Symbol) -> Tuple[sp.Expr, sp.Expr]:
        """Wrapper for `Physics.endpoint`"""
        return self._physics.endpoint(t, self.coordinates)
    
    def COM(self, t: sp.Symbol) -> Tuple[sp.Expr, sp.Expr]:
        """Wrapper for `Physics.COM`"""
        return self._physics.COM(t, self.coordinates)

    def U(self, t: sp.Symbol) -> sp.Expr:
        """Implementation of superclass method
        Wrapper for `Physics.U`
        """
        return self._physics.U(t, self.coordinates)
    
    def T(self, t: sp.Symbol) -> sp.Expr:
        """Implementation of superclass method
        Wrapper for `Physics.T`
        """
        return self._physics.T(t, self.coordinates)

class MultiPendulum(LagrangianBody):
    """
    TODO
    """

    def __init__(self, this: SinglePendulum, constraints: List[Constraint] = None):
        self._this = this
        self._constraints = constraints

        self._next: MultiPendulum = None

    @property
    def this(self) -> SinglePendulum: return self._this
    @property
    def constraints(self) -> List[Constraint]: return self._constraints

    @property
    def next(self) -> MultiPendulum: return self._next
        
    def endpoint(self, t: sp.Symbol) -> Tuple[sp.Expr, sp.Expr]:
        """Returns the endpoint of the final pendulum in this MultiPendulum"""
        if (self.next is not None):
            return self.next.endpoint(t)
        else:
            return self.this.endpoint(t)
    
    def this_DoF(self) -> List[sp.Function]:
        """Returns a list of only those coordinates that are unconstrained degrees of freedom"""
        DoF = self.this.DoF
        if (self.constraints is not None):
            DoF = unconstrained_DoF(DoF, self.constraints)
        return DoF

    @property
    def DoF(self) -> List[sp.Function]:
        """Implementation of superclass method"""
        DoF = self.this_DoF() # Only include unconstrained degrees of freedom
        if (self.next is not None):
            DoF += self.next.DoF
        return DoF
    
    def U(self, t: sp.Symbol) -> sp.Expr:
        """Implementation of superclass method"""
        U = self.this.U(t)
        if (self.next is not None):
            U += self.next.U(t)
        return U
    
    def T(self, t: sp.Symbol) -> sp.Expr:
        """Implementation of superclass method"""
        T = self.this.T(t)
        if (self.next is not None):
            T += self.next.T(t)
        return T
    
    def attach_pendulum(self, t: sp.Symbol, coordinates: SinglePendulum.Coordinates, physics: SinglePendulum.Physics) -> "MultiPendulum":
        """Constructs another MultiPendulum attached to the endpoint of this MultiPendulum
        
        Returns:
            self (to allow method call chaining)
        """

        # Get the endpoint of this pendulum
        x_end, y_end = self.endpoint(t)

        # Setup the constraints
        x_constraint = Constraint(coordinates.x, x_end)
        y_constraint = Constraint(coordinates.y, y_end)

        # Construct the new SinglePendulum
        single_pendulum_new = SinglePendulum(coordinates, physics)

        # Construct the new MultiPendulum to hold the SinglePendulum
        pendulum_new = MultiPendulum(single_pendulum_new, [x_constraint, y_constraint])

        # Attach it to the current pendulum
        self._next = pendulum_new

        # Return the self to allow chaining method calls
        return self

###################################################################################################################################################################################
# SPECIFIC PENDULUM IMPLEMENTATIONS
###################################################################################################################################################################################

class CompoundPendulum(SinglePendulum.Physics):
    """Implementation of SinglePendulum for a single compound pendulum (a pendulum whose mass is evenly distributed along its length)

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

    def endpoint(self, t: sp.Symbol, coordinates: SinglePendulum.Coordinates) -> Tuple[sp.Expr, sp.Expr]:
        """Implementation of superclass method"""
        L = self.L
        x = coordinates.x
        y = coordinates.y
        theta = coordinates.theta

        x_end = x(t) + L * sp.sin(theta(t))
        y_end = y(t) + L * sp.cos(theta(t))

        return (x_end, y_end)

    def COM(self, t: sp.Symbol, coordinates: SinglePendulum.Coordinates) -> Tuple[sp.Expr, sp.Expr]:
        """Implementation of superclass method"""
        L = self.L
        x = coordinates.x
        y = coordinates.y
        theta = coordinates.theta

        x_com = x(t) + L/2 * sp.sin(theta(t))
        y_com = y(t) + L/2 * sp.cos(theta(t))

        return (x_com, y_com)

    def U(self, t: sp.Symbol, coordinates: SinglePendulum.Coordinates) -> sp.Expr:
        """Implementation of superclass method"""
        m = self.m
        _, y_COM = self.COM(t, coordinates)

        g = scipy.constants.g

        return m * g * y_COM

    def T(self, t: sp.Symbol, coordinates: SinglePendulum.Coordinates) -> sp.Expr:
        """Implementation of superclass method"""
        m = self.m
        I = self.I
        theta = coordinates.theta(t)
        x_COM, y_COM = self.COM(t, coordinates)

        x_dot = sp.diff(x_COM, t)
        y_dot = sp.diff(y_COM, t)
        theta_dot = sp.diff(theta, t)

        T_translation = 1/2*m * (x_dot**2 + y_dot**2)
        T_rotation = 1/2*I * theta_dot**2

        return T_translation + T_rotation

###################################################################################################################################################################################
# BUILDERS
###################################################################################################################################################################################

def n_link_pendulum(n: int, physics: SinglePendulum.Physics) -> Tuple[MultiPendulum, sp.Symbol, sp.Function, sp.Function, List[sp.Function]]:
    """
    TODO
    """
    t = sp.symbols("t")

    pendulum = None

    xs = []
    ys = []
    thetas = []
    for i in range(n):
        x = sp.Function("x_" + str(i + 1)) # 1-index the xs
        y = sp.Function("y_" + str(i + 1)) # 1-index the yx
        theta = sp.Function("theta_" + str(i + 1)) # 1-index the thetas

        xs.append(x)
        ys.append(y)
        thetas.append(theta)

        single_pendulum = SinglePendulum(SinglePendulum.Coordinates(x, y, theta), physics)

        if (pendulum is None):
            pendulum = MultiPendulum(single_pendulum)
        else:
            pendulum.attach_pendulum(t, theta, physics)
    
    return (pendulum, t, xs[0], ys[0], thetas)