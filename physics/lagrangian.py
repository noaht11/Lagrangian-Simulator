from abc import ABC, abstractmethod
from typing import List, Tuple, Callable

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

def unconstrained_DoF(all_DoF: List[sp.Function], constraints: List[Constraint]):
    free_DoF = []
    for DoF in all_DoF:
        if (not any((lambda constraint: constraint.coordinate == DoF) for constraint in constraints)):
            # Only include if none of the constraint coordinates are equal to the DoF
            free_DoF.append(DoF)
    return free_DoF

class Lagrangian:
    """Represents the Lagrangian for a physical system

    Attributes:
        `L` : symbolic expression for the Lagrangian of the system
    """

    class ODEExpressions:
        """TODO"""

        def __init__(self, t: sp.Symbol, qs: List[sp.Symbol], p_qs: List[sp.Symbol], q_dots: List[sp.Symbol], force_exprs: List[sp.Expr], momentum_exprs: List[sp.Expr], velocity_exprs: List[sp.Expr]):
            assert(len(qs) == len(p_qs))
            assert(len(qs) == len(q_dots))
            assert(len(qs) == len(force_exprs))
            assert(len(qs) == len(momentum_exprs))
            assert(len(qs) == len(velocity_exprs))

            self._t = t

            self._qs = qs
            self._p_qs = p_qs
            self._q_dots = q_dots
            self._force_exprs = force_exprs
            self._momentum_exprs = momentum_exprs
            self._velocity_exprs = velocity_exprs

        @property
        def num_q(self) -> int:
            return len(self._qs)

        def force_lambdas(self) -> List[Callable]:
            return list(map(lambda force_expr: sp.lambdify([self._t] + self._qs + self._q_dots, force_expr), self._force_exprs))
        
        def momentum_lambdas(self) -> List[Callable]:
            return list(map(lambda momentum_expr: sp.lambdify([self._t] + self._qs + self._q_dots, momentum_expr), self._momentum_exprs))
        
        def velocity_lambdas(self) -> List[Callable]:
            return list(map(lambda velocity_expr: sp.lambdify([self._t] + self._qs + self._p_qs, velocity_expr), self._velocity_exprs))

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
            dL_dq = sp.simplify(dL_dq)
            forces.append(dL_dq)

        # Get generalized momenta
        momenta = []
        for q in degrees_of_freedom:
            q_dot = sp.diff(q(t), t)
            dL_dqdot = sp.diff(L, q_dot).doit()
            dL_dqdot = sp.simplify(dL_dqdot)
            momenta.append(dL_dqdot)
        
        return (forces, momenta)
        
    def solve(self, t: sp.Symbol, degrees_of_freedom: List[sp.Function], constraints: List[Constraint]) -> "ODEExpressions":
        """TODO
        """
        # Generate force and momenta expressions
        (forces, momenta) = self.forces_and_momenta(t, degrees_of_freedom, constraints)

        # Generate symbols for coordinate functions
        qs = list(map(lambda q: sp.Symbol(str(q)), degrees_of_freedom))
        p_qs = list(map(lambda q: sp.Symbol("p_" + str(q)), degrees_of_freedom))
        q_dots = list(map(lambda q: sp.Symbol(str(q) + "_dot"), degrees_of_freedom))
        
        # Replace coordinates with the corresponding symbols
        dq_dts = list(map(lambda q: sp.diff(q(t), t), degrees_of_freedom))
        for i in range(len(degrees_of_freedom)):
            for j in range(len(degrees_of_freedom)):
                forces[i] = forces[i].subs(dq_dts[j], q_dots[j])
                forces[i] = forces[i].subs(degrees_of_freedom[j](t), qs[j])
                momenta[i] = momenta[i].subs(dq_dts[j], q_dots[j])
                momenta[i] = momenta[i].subs(degrees_of_freedom[j](t), qs[j])

        # Generate system of equations to solve for velocities
        velocity_eqs = []

        for p_q, momentum in zip(p_qs, momenta):
            velocity_eqs.append(sp.Eq(p_q, momentum))

        # Solve the system of equations to get expressions for the q_dots
        velocity_solutions, = sp.linsolve(velocity_eqs, q_dots)
        velocities = list(velocity_solutions)

        return Lagrangian.ODEExpressions(t, qs, p_qs, q_dots, forces, momenta, velocities)

class LagrangianBody(ABC):

    @abstractmethod
    def DoF(self) -> List[sp.Function]:
        """
        Returns a list of the coordinates that are degrees of freedom of this body
        """
        pass

    @abstractmethod
    def U(self, t: sp.Symbol) -> sp.Expr:
        """Generates a symbolic expression for the potential energy of this body

        The zero of potential energy is taken to be at a y coordinate of 0

        Arguments:
            `t` : a symbol for the time variable

        Returns:
            A symbolic expression for the potential energy of the pendulum
        """
        pass
    
    @abstractmethod
    def T(self, t: sp.Symbol) -> sp.Expr:
        """Generates and returns a symbolic expression for the kinetic energy of this body

        Arguments:
            `t` : a symbol for the time variable

        Returns:
            A symbolic expression for the kinetic energy of the pendulum
        """
        pass
    
    def lagrangian(self, t: sp.Symbol) -> Lagrangian:
        """Generates and returns the (simplified) Lagrangian for this body"""
        return Lagrangian(sp.simplify(self.T(t).doit()) - sp.simplify(self.U(t).doit()))