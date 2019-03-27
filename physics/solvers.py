from typing import List

from physics.lagrangian import Lagrangian, Constraint

import sympy as sp

class NumericSolver:
    pass

class LagrangianSolver:

    def __init__(self, lagrangian: Lagrangian):
        self._lagrangian = lagrangian

    def _generate_momentum_symbols(self, coordinates: List[sp.Function]) -> List[sp.Symbol]:
        p_qs = []

        for q in coordinates:
            p_qs.append(sp.Symbol("p_" + str(q)))
        
        return p_qs
    
    def _generate_momentum_equations(self, t: sp.Symbol, degrees_of_freedom: List[sp.Function], momenta: List[sp.Expr]) -> List[sp.Eq]:
        p_qs = self._generate_momentum_symbols(degrees_of_freedom)

        eqs = []

        for p_q, momentum in zip(p_qs, momenta):
            eqs.append(sp.Eq(p_q, momentum))
        
        return eqs
    
    def _generate_q_dot_expressions(self, t: sp.Symbol, coordinates: List[sp.Function]) -> List[sp.Expr]:
        q_dots = []

        for q in coordinates:
            q_dots.append(sp.diff(q(t), t))
        
        return q_dots
    
    def solve(self, t: sp.Symbol, degrees_of_freedom: List[sp.Function], constraints: List[Constraint]):
        (forces, momenta) = self._lagrangian.forces_and_momenta(t, degrees_of_freedom, constraints)

        p_equations = self._generate_momentum_equations(t, degrees_of_freedom, momenta)
        q_dots = self._generate_q_dot_expressions(t, degrees_of_freedom)

        q_dots_solutions = sp.linsolve(p_equations, q_dots)