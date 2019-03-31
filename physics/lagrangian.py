from abc import ABC, abstractmethod
from typing import List, Tuple, Callable

import sympy as sp

class DegreeOfFreedom(sp.Function):

    def symbol(self) -> sp.Symbol:
        return sp.Symbol(str(self._coordinate))
    
    def velocity_symbol(self) -> sp.Symbol:
        return sp.Symbol(str(self._coordinate) + "_dot")

    def momentum_symbol(self) -> sp.Symbol:
        return sp.Symbol("p_" + str(self._coordinate))

def degrees_of_freedom(*names) -> Tuple[DegreeOfFreedom,...]:
    return tuple(map(lambda name: DegreeOfFreedom(name), names))
    
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

class Lagrangian:
    """Represents the Lagrangian for a physical system

    Attributes:
        `L`   : symbolic expression for the Lagrangian of the system
        `t`   : a symbol for the time variable
        `DoF` : degrees of freedom as a List of symbolic functions representing the coordinates whose forces and momenta should be derived
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

    def unconstrained_DoFs(DoFs: List[DegreeOfFreedom], constraints: List[Constraint]):
        free_DoFs = []
        for DoF in DoFs:
            if (not any((lambda constraint: constraint.coordinate == DoF) for constraint in constraints)):
                # Only include if none of the constraint coordinates are equal to the DoF
                free_DoFs.append(DoF)
        return free_DoFs

    def __init__(self, L: sp.Expr, t: sp.Symbol, DoFs: List[DegreeOfFreedom]):
        self._L = L
        self._t = t
        self._DoFs = DoFs

    def __str__(self):
        return str(self.L)
    
    def print(self):
        """Pretty prints the symbolic expression for this Lagrangian"""
        sp.pprint(self.L)
    
    def forces_and_momenta(self) -> Tuple[List[sp.Expr], List[sp.Expr]]:
        """Calculates the generalized forces and momenta for this Lagrangian for each of the provided degrees of freedom

        If any coordinates are to be constrained by specific values or expressions these constrains should be passed in the `constraints` argument.

        All coordinates in the Lagrangian should appear in EITHER the `degrees_of_freedom` OR the `constraints`, but NOT BOTH.
        
        Returns:
            Tuple of the form (`forces`, `momenta`):
                `forces` is a List of generalized forces (where each force is a symbolic expression and corresponds to the coordinate at the same index in the `degrees_of_freedom` list)
                `momenta` is a List of generalized momenta (each momentum is a symbolic expression and corresponds to the coordinate at the same index in the `degrees_of_freedom` list)
        """
        # Convenience variable for Lagrangian expression
        L = self._L
        t = self._t
        DoFs = self._DoFs

        # Get generalized forces
        forces = []
        for q in DoFs:
            dL_dq = sp.diff(L, q(t)).doit()
            dL_dq = sp.simplify(dL_dq)
            forces.append(dL_dq)

        # Get generalized momenta
        momenta = []
        for q in DoFs:
            q_dot = sp.diff(q(t), t)
            dL_dqdot = sp.diff(L, q_dot).doit()
            dL_dqdot = sp.simplify(dL_dqdot)
            momenta.append(dL_dqdot)
        
        return (forces, momenta)
        
    def solve(self) -> "ODEExpressions":
        """TODO"""
        L = self._L
        t = self._t
        DoFs = self._DoFs

        # Generate force and momenta expressions
        (forces, momenta) = self.forces_and_momenta()

        # Generate symbols for coordinate functions
        qs = list(map(lambda q: q.symbol(), DoFs))
        p_qs = list(map(lambda q: q.momentum_symbol(), DoFs))
        q_dots = list(map(lambda q: q.velocity_symbol(), DoFs))
        
        # Replace coordinates with the corresponding symbols
        dq_dts = list(map(lambda q: sp.diff(q(t), t), DoFs))
        for i in range(len(DoFs)):
            for j in range(len(DoFs)):
                forces[i] = forces[i].subs(dq_dts[j], q_dots[j])
                forces[i] = forces[i].subs(DoFs[j](t), qs[j])
                momenta[i] = momenta[i].subs(dq_dts[j], q_dots[j])
                momenta[i] = momenta[i].subs(DoFs[j](t), qs[j])

        # Generate system of equations to solve for velocities
        velocity_eqs = []

        for p_q, momentum in zip(p_qs, momenta):
            velocity_eqs.append(sp.Eq(p_q, momentum))

        # Solve the system of equations to get expressions for the velocities
        velocity_solutions, = sp.linsolve(velocity_eqs, q_dots)
        velocities = list(velocity_solutions)

        return Lagrangian.ODEExpressions(t, qs, p_qs, q_dots, forces, momenta, velocities)

class LagrangianBody:

    class LagrangianPhysics(ABC):
        @abstractmethod
        def DoFs(self) -> List[DegreeOfFreedom]:
            """See LagrangianBody.DoFs"""
            pass

        @abstractmethod
        def U(self, t: sp.Symbol) -> sp.Expr:
            """See LagrangianBody.U"""
            pass
        
        @abstractmethod
        def T(self, t: sp.Symbol) -> sp.Expr:
            """See LagrangianBody.T"""
            pass

    def __init__(self, physics: LagrangianPhysics, t: sp.Symbol, *constraints: Constraint):
        self._physics = physics
        self._t = t
        self._constraints = constraints

    def DoFs(self) -> List[DegreeOfFreedom]:
        """
        Returns a list of the coordinates that are degrees of freedom of this body
        """
        return LagrangianBody.unconstrained_DoF(self._physics.DoF(), self._constraints)

    @abstractmethod
    def U(self) -> sp.Expr:
        """Generates a symbolic expression for the potential energy of this body

        The zero of potential energy is taken to be at a y coordinate of 0

        Arguments:
            `t` : a symbol for the time variable

        Returns:
            A symbolic expression for the potential energy of the pendulum
        """
        t = self._t
        U = self._physics.U(t)

        for constraint in self._constraints:
            constraint.apply_to(t, U)
        
        return U.simplify()
    
    @abstractmethod
    def T(self) -> sp.Expr:
        """Generates and returns a symbolic expression for the kinetic energy of this body

        Arguments:
            `t` : a symbol for the time variable

        Returns:
            A symbolic expression for the kinetic energy of the pendulum
        """
        t = self._t
        T = self._physics.T(t)

        for constraint in self._constraints:
            constraint.apply_to(t, T)
        
        return T.simplify()
    
    def lagrangian(self) -> Lagrangian:
        """Generates and returns the (simplified) Lagrangian for this body"""
        return Lagrangian(sp.simplify(self.T(self._t).doit()) - sp.simplify(self.U(self._t).doit()), self._t, self.DoF())
    
    def constrain(self, *constraints: Constraint):
        return LagrangianBody(self._physics, self._t, constraints)