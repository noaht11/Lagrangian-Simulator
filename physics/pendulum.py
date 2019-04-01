from typing import List, Tuple, Dict, Callable
from abc import ABC, abstractmethod
from math import pi

import sympy as sp
import numpy as np
import scipy.constants

from physics.lagrangian import Lagrangian, LagrangianBody, DegreeOfFreedom, Constraint
from physics.animation import Artist
from physics.solver import Solver

###################################################################################################################################################################################
# UTILITY FUNCTIONS
###################################################################################################################################################################################

def neg_pi_to_pi(theta: float) -> float:
    """Converts an angle in radians to the equivalent angle in radians constrained between -pi and pi, where 0 remains at angle 0
    """
    modded = theta % (2*pi)
    return modded + (modded > pi) * (-2*pi)

###################################################################################################################################################################################
# LAGRANGIAN CLASSES
###################################################################################################################################################################################

class SinglePendulumLagrangianPhysics(LagrangianBody.LagrangianPhysics):
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
    
    class PendulumCoordinates:
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

    class PendulumPhysics(ABC):
        """Abstract base class for representing the physical properties and behavior of the pendulum
        
        This class defines the following abstract methods as they are considered to exist for all pendulums, but their implementation depends on the type of pendulum:

            - `endpoint` : returns the location of the endpoint
            - `COM`      : returns the location of the center of mass
            - `U`        : returns the potential energy
            - `T`        : returns the kinetic energy

        See the documentation of these methods for more information.
        """

        @abstractmethod
        def endpoint(self, t: sp.Symbol, coordinates: "PendulumCoordinates") -> Tuple[sp.Expr, sp.Expr]:
            """See `SinglePendulumLagrangianPhysics.endpoint`"""
            pass

        @abstractmethod
        def COM(self, t: sp.Symbol, coordinates: "PendulumCoordinates") -> Tuple[sp.Expr, sp.Expr]:
            """See `SinglePendulumLagrangianPhysics.COM`"""
            pass

        @abstractmethod
        def U(self, t: sp.Symbol, coordinates: "PendulumCoordinates") -> sp.Expr:
            """See `SinglePendulumLagrangianPhysics.U`"""
            pass
        
        @abstractmethod
        def T(self, t: sp.Symbol, coordinates: "PendulumCoordinates") -> sp.Expr:
            """See `SinglePendulumLagrangianPhysics.T`"""
            pass

    def __init__(self, coordinates: PendulumCoordinates, physics: PendulumPhysics, DoFs: List[DegreeOfFreedom]):
        self._coordinates = coordinates
        self._physics = physics
        self._DoFs = DoFs
    
    @property
    def coordinates(self) -> PendulumCoordinates : return self._coordinates
    
    def pivot(self, t: sp.Symbol) -> Tuple[sp.Expr, sp.Expr]:
        return (self.coordinates.x(t), self.coordinates.y(t))

    def endpoint(self, t: sp.Symbol) -> Tuple[sp.Expr, sp.Expr]:
        """Generates symbolic expressions for the coordinates of the endpoint of the pendulum

        The endpoint of the pendulum is the point where a subsequent pendulum would be attached if a chain of pendulums were to be constructed.

        Arguments:
            `t` : a symbol for the time variable

        Returns:
            Tuple of (x_end, y_end) where each coordinate is a symbolic expression
        """
        return self._physics.endpoint(t, self.coordinates)
    
    def COM(self, t: sp.Symbol) -> Tuple[sp.Expr, sp.Expr]:
        """Generates symbolic expressions for the coordinates of the centre of mass (COM) of the pendulum

        Arguments:
            `t` : a symbol for the time variable

        Returns:
            Tuple of (x_COM, y_COM) where each coordinate is a symbolic expression
        """
        return self._physics.COM(t, self.coordinates)

    def DoFs(self) -> List[DegreeOfFreedom]:
        """Implementation of superclass method"""
        return self._DoFs

    def U(self, t: sp.Symbol) -> sp.Expr:
        """Implementation of superclass method"""
        return self._physics.U(t, self.coordinates)
    
    def T(self, t: sp.Symbol) -> sp.Expr:
        """Implementation of superclass method"""
        return self._physics.T(t, self.coordinates)

class MultiPendulumLagrangianPhysics(LagrangianBody.LagrangianPhysics):
    """
    TODO mutable or not mutable (_next)?
    """

    def __init__(self, this: SinglePendulumLagrangianPhysics):
        self._this = this
        self._next: MultiPendulumLagrangianPhysics = None

    @property
    def this(self) -> SinglePendulumLagrangianPhysics: return self._this
    
    @property
    def next(self) -> "MultiPendulumLagrangianPhysics": return self._next
        
    def endpoint(self, t: sp.Symbol) -> Tuple[sp.Expr, sp.Expr]:
        """Returns the endpoint of the final pendulum in this MultiPendulumLagrangianPhysics"""
        if (self.next is not None):
            return self.next.endpoint(t)
        else:
            return self.this.endpoint(t)
    
    def DoFs(self) -> List[DegreeOfFreedom]:
        """Implementation of superclass method"""
        DoFs = self.this.DoFs()
        if (self.next is not None):
            DoFs += self.next.DoFs()
        return DoFs
    
    def U(self, t: sp.Symbol) -> sp.Expr:
        """Implementation of superclass method"""
        # Potential energy of this
        U = self.this.U(t)

        # Potential energy of next
        if (self.next is not None):
            U += self.next.U(t)

        return U
    
    def T(self, t: sp.Symbol) -> sp.Expr:
        """Implementation of superclass method"""
        # Kinetic energy of this
        T = self.this.T(t)
        
        # Kinetic energy of next
        if (self.next is not None):
            T += self.next.T(t)

        return T
    
    def attach_pendulum(self, t: sp.Symbol, theta: DegreeOfFreedom, physics: SinglePendulumLagrangianPhysics.PendulumPhysics) -> "MultiPendulumLagrangianPhysics":
        """Constructs another MultiPendulumLagrangianPhysics attached to the endpoint of this MultiPendulumLagrangianPhysics
        
        Returns:
            The newly added MultiPendulumLagrangianPhysics (to allow method call chaining)
        """
        # Attach to the last pendulum in the chain
        if (self.next is not None):
            return self.next.attach_pendulum(t, physics)

        # Get the endpoint of this pendulum
        x_end, y_end = self.this.endpoint(t)
        coordinates = SinglePendulumLagrangianPhysics.PendulumCoordinates(sp.Lambda(t, x_end), sp.Lambda(t, y_end), theta)

        # Create Single Pendulum
        pendulum = SinglePendulumLagrangianPhysics(coordinates, physics, [theta])

        # Construct the new MultiPendulumLagrangianPhysics to hold the SinglePendulumLagrangianPhysics
        pendulum_new = MultiPendulumLagrangianPhysics(pendulum)

        # Attach it to the current pendulum
        self._next = pendulum_new

        # Return the new pendulum to allow chaining method calls
        return pendulum_new

###################################################################################################################################################################################
# SPECIFIC LAGRANGIAN PENDULUM IMPLEMENTATIONS
###################################################################################################################################################################################

class CompoundPendulumPhysics(SinglePendulumLagrangianPhysics.PendulumPhysics):
    """Implementation of SinglePendulumLagrangianPhysics for a single compound pendulum (a pendulum whose mass is evenly distributed along its length)

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

    def endpoint(self, t: sp.Symbol, coordinates: SinglePendulumLagrangianPhysics.PendulumCoordinates) -> Tuple[sp.Expr, sp.Expr]:
        """Implementation of superclass method"""
        L = self.L
        x = coordinates.x
        y = coordinates.y
        theta = coordinates.theta

        x_end = x(t) + L * sp.sin(theta(t))
        y_end = y(t) + L * sp.cos(theta(t))

        return (x_end, y_end)

    def COM(self, t: sp.Symbol, coordinates: SinglePendulumLagrangianPhysics.PendulumCoordinates) -> Tuple[sp.Expr, sp.Expr]:
        """Implementation of superclass method"""
        L = self.L
        x = coordinates.x
        y = coordinates.y
        theta = coordinates.theta

        x_com = x(t) + L/2 * sp.sin(theta(t))
        y_com = y(t) + L/2 * sp.cos(theta(t))

        return (x_com, y_com)

    def U(self, t: sp.Symbol, coordinates: SinglePendulumLagrangianPhysics.PendulumCoordinates) -> sp.Expr:
        """Implementation of superclass method"""
        m = self.m
        _, y_COM = self.COM(t, coordinates)

        g = scipy.constants.g

        return m * g * y_COM

    def T(self, t: sp.Symbol, coordinates: SinglePendulumLagrangianPhysics.PendulumCoordinates) -> sp.Expr:
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
# LAGRANGIAN BODY CLASSES
###################################################################################################################################################################################

class SinglePendulum(LagrangianBody):

    def __init__(self, t: sp.Symbol, physics: SinglePendulumLagrangianPhysics, *constraints: Constraint):
        super().__init__(t, physics, *constraints)
    
        self._pendulum_physics = physics
    
    def pivot(self) -> Tuple[sp.Expr, sp.Expr]:
        t = self.t
        x = self._pendulum_physics.coordinates.x(t)
        y = self._pendulum_physics.coordinates.y(t)
        return (x, y)
    
    def endpoint(self) -> Tuple[sp.Expr, sp.Expr]:
        t = self.t
        endpoint = self._pendulum_physics.endpoint(t)
        return endpoint

class MultiPendulum(LagrangianBody):
    def __init__(self, t: sp.Symbol, physics: MultiPendulumLagrangianPhysics, *constraints: Constraint):
        super().__init__(t, physics, *constraints)
    
        self._pendulum_physics = physics
    
    @property
    def physics(self) -> MultiPendulumLagrangianPhysics: return self._pendulum_physics


###################################################################################################################################################################################
# ANIMATION CLASSES
###################################################################################################################################################################################

class SinglePendulumArtist(Artist):

    def __init__(self, x: Callable[..., float], y: Callable[..., float], endpoint_x: Callable[..., float], endpoint_y: Callable[..., float]):
        self._x = x
        self._y = y
        self._endpoint_x = endpoint_x
        self._endpoint_y = endpoint_y
    
    def reset(self, axes):
        self._line, = axes.plot([], [], '-', lw=4)
        return self._line,

    def draw(self, state: np.ndarray):
        x = self._x(*state)
        y = self._y(*state)
        endpoint_x = self._endpoint_x(*state)
        endpoint_y = self._endpoint_y(*state)

        x_data = [x, endpoint_x]
        y_data = [y, endpoint_y]
        self._line.set_data(x_data, y_data)

        return self._line,

class MultiPendulumArtist(Artist):

    def __init__(self, this: SinglePendulumArtist, next: "MultiPendulumArtist"):
        self._this = this
        self._next = next

    def reset(self, axes):
        this_mod = self._this.reset(axes)
        
        if (self._next is not None):
            next_mod = self._next.reset(axes)
            return this_mod, next_mod

        return this_mod,

    def draw(self, state: np.ndarray):
        this_mod = self._this.draw(state)
        
        if (self._next is not None):
            next_mod = self._next.draw(state)
            return this_mod, next_mod

        return this_mod,

###################################################################################################################################################################################
# SOLVER CLASSES
###################################################################################################################################################################################

class SinglePendulumSolver(Solver):

    def __init__(self, single_pendulum: SinglePendulum):
        super().__init__(single_pendulum)
        self._single_pendulum = single_pendulum
    
    def artist(self) -> SinglePendulumArtist:
        (x, y) = self._single_pendulum.pivot()
        (endpoint_x, endpoint_y) = self._single_pendulum.endpoint()

        t = self._single_pendulum.t
        DoFs = self._single_pendulum.DoFs()

        exprs = [x, y, endpoint_x, endpoint_y]
        exprs = list(map(lambda expr: Lagrangian.apply_constraints(t, expr, self._single_pendulum.constraints), exprs))

        (exprs, qs, q_dots) = Lagrangian.symbolize(exprs, t, DoFs)

        exprs_lambdas = list(map(lambda expr: sp.lambdify(qs + q_dots, expr), exprs))

        return SinglePendulumArtist(*exprs_lambdas)

class MultiPendulumSolver(Solver):

    def __init__(self, multi_pendulum: MultiPendulum):
        super().__init__(multi_pendulum)
        self._multi_pendulum = multi_pendulum
    
    def _create_artist(multi_pendulum_physics: MultiPendulumLagrangianPhysics) -> MultiPendulumArtist:
        multi_pendulum_physics.this

        this: SinglePendulumLagrangianPhysics = multi_pendulum_physics.this

    def artist(self) -> MultiPendulumArtist:
        pass

###################################################################################################################################################################################
# HELPERS
###################################################################################################################################################################################

def n_link_pendulum(n: int, physics: SinglePendulumLagrangianPhysics.PendulumPhysics) -> Tuple[MultiPendulumLagrangianPhysics, sp.Symbol, DegreeOfFreedom, DegreeOfFreedom, List[DegreeOfFreedom]]:
    """
    TODO
    """
    if (n < 1):
        raise AssertionError("n must be at least 1")

    t = sp.Symbol("t")

    root_pendulum = None
    last_pendulum = None

    thetas = []
    for i in range(n):
        theta = DegreeOfFreedom("theta_" + str(i + 1)) # 1-index the thetas

        thetas.append(theta)

        single_pendulum = SinglePendulumLagrangianPhysics(SinglePendulumLagrangianPhysics.PendulumCoordinates(x, y, theta), physics)

        if (root_pendulum is None):
            root_pendulum = MultiPendulumLagrangianPhysics(single_pendulum)
            last_pendulum = root_pendulum
        else:
            last_pendulum = last_pendulum.attach_pendulum(t, single_pendulum)
    
    return (root_pendulum, t, xs[0], ys[0], thetas)