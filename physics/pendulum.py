from typing import List, Tuple, Dict, Callable
from abc import ABC, abstractmethod
from math import pi

import sympy as sp
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

        def __init__(self, x: DegreeOfFreedom, y: DegreeOfFreedom, theta: DegreeOfFreedom):
            self._x = x
            self._y = y
            self._theta = theta
        
        @property
        def x(self)     -> DegreeOfFreedom : return self._x
        @property
        def y(self)     -> DegreeOfFreedom : return self._y
        @property
        def theta(self) -> DegreeOfFreedom : return self._theta

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

    def __init__(self, coordinates: PendulumCoordinates, physics: PendulumPhysics):
        self._coordinates = coordinates
        self._physics = physics
    
    @property
    def coordinates(self) -> PendulumCoordinates : return self._coordinates

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

    def DoFs(self) -> List[sp.Function]:
        """Implementation of superclass method"""
        return [self.coordinates.x, self.coordinates.y, self.coordinates.theta]

    def U(self, t: sp.Symbol) -> sp.Expr:
        """Implementation of superclass method"""
        return self._physics.U(t, self.coordinates)
    
    def T(self, t: sp.Symbol) -> sp.Expr:
        """Implementation of superclass method"""
        return self._physics.T(t, self.coordinates)

class MultiPendulumLagrangianPhysics(LagrangianBody.LagrangianPhysics):
    """
    TODO
    """

    def __init__(self, this: SinglePendulumLagrangianPhysics, constraints: List[Constraint] = None):
        self._this = this
        self._constraints = constraints

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
    
    def _this_DoFs(self) -> List[DegreeOfFreedom]:
        """Returns a list of only those coordinates that are unconstrained degrees of freedom"""
        DoFs = self._this.DoFs()
        if (self._constraints is not None):
            DoFs = Lagrangian.unconstrained_DoFs(DoFs, self._constraints)
        return DoFs
    
    def _apply_constraints(self, t: sp.Symbol, expression: sp.Expr) -> sp.Expr:
        constrained = expression
        if self._constraints is not None:
            for constraint in self._constraints:
                constrained = constraint.apply_to(t, constrained)
        return constrained

    def DoFs(self) -> List[DegreeOfFreedom]:
        """Implementation of superclass method"""
        DoFs = self._this_DoFs() # Only include unconstrained degrees of freedom
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

        # Apply constraints    
        U = self._apply_constraints(t, U)

        return U
    
    def T(self, t: sp.Symbol) -> sp.Expr:
        """Implementation of superclass method"""
        # Kinetic energy of this
        T = self.this.T(t)
        
        # Kinetic energy of next
        if (self.next is not None):
            T += self.next.T(t)
        
        # Apply constraints
        T = self._apply_constraints(t, T)

        return T
    
    def attach_pendulum(self, t: sp.Symbol, pendulum: SinglePendulumLagrangianPhysics) -> "MultiPendulumLagrangianPhysics":
        """Constructs another MultiPendulumLagrangianPhysics attached to the endpoint of this MultiPendulumLagrangianPhysics
        
        Returns:
            The newly added MultiPendulumLagrangianPhysics (to allow method call chaining)
        """
        # Attach to the last pendulum in the chain
        if (self.next is not None):
            return self.next.attach_pendulum(t, pendulum)

        # Get the endpoint of this pendulum
        x_end, y_end = self.endpoint(t)

        # Setup the constraints
        x_constraint = Constraint(pendulum.coordinates.x, x_end)
        y_constraint = Constraint(pendulum.coordinates.y, y_end)

        # Construct the new MultiPendulumLagrangianPhysics to hold the SinglePendulumLagrangianPhysics
        pendulum_new = MultiPendulumLagrangianPhysics(pendulum, [x_constraint, y_constraint])

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
        return self._line

    def draw(self, state: List[float]):
        x = self._x(*state)
        y = self._y(*state)
        endpoint_x = self._endpoint_x(*state)
        endpoint_y = self._endpoint_y(*state)

        x_data = [x, endpoint_x]
        y_data = [y, endpoint_y]
        self._line.set_data(x_data, y_data)

        return self._line

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
        for constraint in self._single_pendulum.constraints:
            exprs = list(map(lambda expr: constraint.apply_to(t, expr), exprs))

        (exprs, qs, q_dots) = Lagrangian.symbolize(exprs, t, DoFs)

        exprs_lambdas = list(map(lambda expr: sp.lambdify(qs + q_dots, expr), exprs))

        return SinglePendulumArtist(*exprs_lambdas)

###################################################################################################################################################################################
# HELPERS
###################################################################################################################################################################################

def n_link_pendulum(n: int, physics: SinglePendulumLagrangianPhysics.PendulumPhysics) -> Tuple[MultiPendulumLagrangianPhysics, sp.Symbol, DegreeOfFreedom, DegreeOfFreedom, List[DegreeOfFreedom]]:
    """
    TODO
    """
    t = sp.Symbol("t")

    root_pendulum = None
    last_pendulum = None

    xs = []
    ys = []
    thetas = []
    for i in range(n):
        x = DegreeOfFreedom("x_" + str(i + 1)) # 1-index the xs
        y = DegreeOfFreedom("y_" + str(i + 1)) # 1-index the yx
        theta = DegreeOfFreedom("theta_" + str(i + 1)) # 1-index the thetas

        xs.append(x)
        ys.append(y)
        thetas.append(theta)

        single_pendulum = SinglePendulumLagrangianPhysics(SinglePendulumLagrangianPhysics.PendulumCoordinates(x, y, theta), physics)

        if (root_pendulum is None):
            root_pendulum = MultiPendulumLagrangianPhysics(single_pendulum)
            last_pendulum = root_pendulum
        else:
            last_pendulum = last_pendulum.attach_pendulum(t, single_pendulum)
    
    return (root_pendulum, t, xs[0], ys[0], thetas)