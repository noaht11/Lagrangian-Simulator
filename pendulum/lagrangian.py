from typing import List, Tuple

import sympy as sp

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
        """Pretty prints the symbolic expression for this Lagrangian
        """
        sp.pprint(self.T - self.U)

    def generalized_forces_momenta(self, t: sp.Symbol, coordinates: List[sp.Function], substitutions: List[Tuple[sp.Function, sp.Expr]]) -> Tuple[List[sp.Expr], List[sp.Expr]]:
        """Calculates the generalized forces and momenta for this Lagrangian for each of the provided coordinates

        If any functions within the Lagrangian are pre-determined they can be replaced with explicit expressions prior to generating
        the force and momenta equations

        Arguments:
            `t`             : a symbol for the time variable
            `coordinates`   : a List of symbolic functions represented the coordinates whose forces and momenta should be derived
            `substitutions` : a List of Tuples of the form (`function`, `expression`) indicating substitutions to make before calculating the forces and momenta.
                              For each substitution, every instance of `function` within the Lagrangian will be replaced by `expression`
        
        Returns:
            Tuple of the form (`forces`, `momenta`):
                `forces` is a List of generalized forces (where each force is a symbolic expression and corresponds to the coordinate at the same index in the coordinate list)
                `momenta` is a List of generalized momenta (each momentum is a symbolic expression and corresponds to the coordinate at the same index in the coordinate list)
        """
        # Define the Lagrangian
        L = self.T - self.U
        
        # Substitute in pre-determined functions
        for func, expr in substitutions:
            L = L.subs(func(t), expr)

        # Simplify the final Lagrangian
        L = sp.simplify(L)

        # Get generalized forces
        forces = []
        for q in coordinates:
            dL_dq = sp.diff(L, q(t)).doit()
            forces.append(dL_dq)

        # Get generalized momenta
        momenta = []
        for q in coordinates:
            q_dot = sp.diff(q(t), t)
            dL_dqdot = sp.diff(L, q_dot).doit()
            momenta.append(dL_dqdot)
        
        return (forces, momenta)
