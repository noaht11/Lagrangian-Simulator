from abc import ABC, abstractmethod
from typing import List, Tuple, Callable, Dict

import sympy as sp

from pendulum.test import *

class Lagrangian():
    """Represents the Lagrangian for a physical system

    Attributes:
        `U` : symbolic expression for the potential energy of the system
        `T` : symbolic expression for the kinetic energy of the system
    """

    def __init__(self, U: sp.Expr, T: sp.Expr):
        self._U = U
        self._T = T
    
    @property
    def U(self) -> sp.Expr: return self._U
    @property
    def T(self) -> sp.Expr: return self._T

    def __str__(self):
        return str(self.T - self.U)
    
    def print(self):
        sp.pprint(self.T - self.U)

    def generalized_forces_momenta(self, t: sp.Symbol, substitutions: List[Tuple[sp.Function, sp.Expr]], coordinates: List[sp.Expr]):
        """
        """
        # Define the Lagrangian
        L = self.T - self.U

        # Substitute in any functions that won't be degrees of freedom
        for func, expr in substitutions:
            L = L.subs(func(t), expr)

        # Simplify the final Lagrangian
        L = sp.simplify(L)

        # Get generalized forces
        gen_forces = []
        for q in coordinates:
            dL_dq = sp.diff(L, q).doit()
            gen_forces.append(dL_dq)

        # Get generalized momenta
        gen_momenta = []
        for q in coordinates:
            q_dot = sp.diff(q, t).doit()
            dL_dqdot = sp.diff(L, q_dot).doit()
            gen_momenta.append(dL_dqdot)
        
        return (gen_forces, gen_momenta)
