from abc import ABC, abstractmethod
from typing import List, Tuple

import sympy as sp

class Constraint:

    def __init__(self, coordinate: sp.Function, expression: sp.Expr):
        self._coordinate = coordinate
        self._expression = expression
    
    @property
    def coordinate(self) -> sp.Function: return self._coordinate
    @property
    def expression(self) -> sp.Expr: return self._expression

    def apply_to(self, t: sp.Symbol, E: sp.Expr) -> sp.Expr:
        """Applies this constraint to the provided expression and returns the newly constrained expression"""
        return E.subs(self.coordinate(t), self.expression)


class Lagrangian():
    """Represents the Lagrangian for a physical system

    Attributes:
        `L` : symbolic expression for the Lagrangian of the system
    """

    def __init__(self, L: sp.Expr):
        self._L = L
    
    @property
    def L(self) -> sp.Expr: return self._L

    def __str__(self):
        return str(self.L)
    
    def print(self):
        """Pretty prints the symbolic expression for this Lagrangian"""
        sp.pprint(self.L)
    
    def forces_and_momenta(self, t: sp.Symbol, degrees_of_freedom: List[sp.Function], constraints: List[Constraint]) -> Tuple[List[sp.Expr], List[sp.Expr]]:
        """Calculates the generalized forces and momenta for this Lagrangian for each of the provided degrees of freedom

        If any coordinates are to be constrained by specific values or expressions these constrains should be passed in the `constraints` argument.

        All coordinates in the Lagrangian should appear in EITHER the `degrees_of_freedom` OR the `constraints`, but NOT BOTH.

        Arguments:
            `t`                  : a symbol for the time variable
            `degrees_of_freedom` : a List of symbolic functions representing the coordinates whose forces and momenta should be derived
            `constraints`        : a List of Constraint objects indicating coordinates that should be replaced with explicit expressions prior to calculating forces and momenta.
        
        Returns:
            Tuple of the form (`forces`, `momenta`):
                `forces` is a List of generalized forces (where each force is a symbolic expression and corresponds to the coordinate at the same index in the `degrees_of_freedom` list)
                `momenta` is a List of generalized momenta (each momentum is a symbolic expression and corresponds to the coordinate at the same index in the `degrees_of_freedom` list)
        """
        # Convenience variable for Lagrangian expression
        L = self.L
        
        # Substitute in constraints
        for constraint in constraints:
            L = constraint.apply_to(t, L)

        # Simplify the final Lagrangian
        L = sp.simplify(L)

        # Get generalized forces
        forces = []
        for q in degrees_of_freedom:
            dL_dq = sp.diff(L, q(t)).doit()
            forces.append(dL_dq)

        # Get generalized momenta
        momenta = []
        for q in degrees_of_freedom:
            q_dot = sp.diff(q(t), t)
            dL_dqdot = sp.diff(L, q_dot).doit()
            momenta.append(dL_dqdot)
        
        return (forces, momenta)

class LagrangianBody(ABC):

    @abstractmethod
    @property
    def DoF(self) -> List[sp.Function]:
        """
        TODO
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
    
    def lagrangian(self, t: sp.Symbol) -> Lagrangian:
        """Generates and returns the Lagrangian for this body"""
        return Lagrangian(self.T(t) - self.U(t))