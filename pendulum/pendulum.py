from typing import List, Tuple, Dict
from abc import ABC, abstractmethod
from math import pi

import sympy as sp
import scipy.constants

from pendulum.lagrangian import LagrangianBody, Constraint

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
# CLASSES
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

    @property
    def DoF(self) -> List[sp.Function]:
        """Implementation of superclass method"""
        return [self.x, self.y, self.theta]
    
    @abstractmethod
    def endpoint(self, t: sp.Symbol) -> Tuple[sp.Expr, sp.Expr]:
        """Generates symbolic expressions for the coordinates of the endpoint of the pendulum

        The endpoint of the pendulum is the point where a subsequent pendulum would be attached if a chain of pendulums were to be constructed.

        Arguments:
            `t` : a symbol for the time variable

        Returns:
            Tuple of (x_end, y_end) where each coordinate is a symbolic expression
        """
        pass
    
class MultiPendulum(LagrangianBody):
    """
    TODO
    """

    def __init__(self, this: SinglePendulum, constraints: List[Constraint] = None):
        self._this = this
        self._constraints = constraints

        self._next = None

    @property
    def this(self) -> SinglePendulum: return self._this
    @property
    def constraints(self) -> List[Constraint]: return self._constraints

    @property
    def next(self) -> MultiPendulum: return self._next
    
    def this_DoF(self) -> List[sp.Function]:
        all_DoF = self.this.DoF
        free_DoF = []
        constrained_DoF = []

        if (self.constraints is not None):
            for constraint in self.constraints:
                constrained_DoF.append(constraint.coordinate)

            for DoF in all_DoF:
                if (DoF not in constrained_DoF):
                    free_DoF.append(DoF)
        else:
            free_DoF = all_DoF
        
        return free_DoF

    @property
    def DoF(self) -> List[sp.Function]:
        """Implementation of superclass method"""
        DoF = self.this_DoF()
        if (self.next is not None):
            DoF += self.next.DoF
        return DoF
    
    def U(self, t: sp.Symbol) -> sp.Expr:
        """Implementation of superclass method"""
        U = super().U(t)
        if (self.attachment is not None):
            U += self.attachment.U(t)
        return U
    
    def T(self, t: sp.Symbol) -> sp.Expr:
        """Implementation of superclass method"""
        T = super().T(t)
        if (self.attachment is not None):
            T += self.attachment.T(t)
        return T