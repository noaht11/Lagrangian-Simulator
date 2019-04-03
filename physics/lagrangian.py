from abc import ABC, abstractmethod
from typing import List, Tuple, Callable

import sympy as sp

class DegreeOfFreedom:

    def __init__(self, name: str):
        self._coordinate = sp.Function(name)

    @property
    def coordinate(self) -> sp.Function: return self._coordinate

    def __call__(self, *args):
        return self._coordinate(*args)

    def __str__(self):
        return str(self._coordinate)

    def symbol(self) -> sp.Symbol:
        return sp.Symbol(str(self._coordinate))
    
    def velocity_symbol(self) -> sp.Symbol:
        return sp.Symbol(str(self._coordinate) + "_dot")

    def momentum_symbol(self) -> sp.Symbol:
        return sp.Symbol("p_" + str(self._coordinate))

def degrees_of_freedom(*names) -> Tuple[DegreeOfFreedom,...]:
    return tuple(map(lambda name: DegreeOfFreedom(name), names))
    
class Constraint:

    def __init__(self, DoF: DegreeOfFreedom, expression: sp.Expr):
        self._DoF = DoF
        self._expression = expression
    
    @property
    def DoF(self) -> DegreeOfFreedom: return self._DoF
    @property
    def expression(self) -> sp.Expr: return self._expression

    def __str__(self):
        # TODO also include expression
        return str(self._DoF)

    def apply_to(self, t: sp.Symbol, E: sp.Expr) -> sp.Expr:
        """Applies this constraint to the provided expression and returns the newly constrained expression"""
        return E.subs(self.DoF(t), self.expression)

class Lagrangian:
    """Represents the Lagrangian for a physical system

    Attributes:
        'L'   : symbolic expression for the Lagrangian of the system
        't'   : a symbol for the time variable
        'DoF' : degrees of freedom as a List of symbolic functions representing the coordinates whose forces and momenta should be derived
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
        def t(self) -> sp.Symbol: return self._t

        @property
        def num_q(self) -> int: return len(self._qs)
        @property
        def qs(self) -> List[sp.Symbol]: return self._qs
        @property
        def p_qs(self) -> List[sp.Symbol]: return self._p_qs
        @property
        def q_dots(self) -> List[sp.Symbol]: return self._q_dots
        @property
        def force_exprs(self) -> List[sp.Symbol]: return self._force_exprs
        @property
        def momentum_exprs(self) -> List[sp.Symbol]: return self._momentum_exprs
        @property
        def velocity_exprs(self) -> List[sp.Symbol]: return self._velocity_exprs

    @staticmethod
    def same_coordinate(a: sp.Function, b: sp.Function) -> bool:
        return str(a) == str(b)

    @staticmethod
    def unconstrained_DoFs(DoFs: List[DegreeOfFreedom], constraints: List[Constraint]) -> List[DegreeOfFreedom]:
        free_DoFs = []
        for DoF in DoFs:
            constrained = False
            for constraint in constraints:
                if (Lagrangian.same_coordinate(constraint.DoF.coordinate, DoF.coordinate)):
                    constrained = True
            if not constrained: free_DoFs.append(DoF)
        return free_DoFs
    
    @staticmethod
    def apply_constraints(t: sp.Symbol, expr: sp.Expr, constraints: List[Constraint]) -> sp.Expr:
        for constraint in constraints:
            expr = constraint.apply_to(t, expr)
        return expr

    @staticmethod
    def symbolize(expressions: List[sp.Expr], t: sp.Symbol, DoFs: List[DegreeOfFreedom]) -> Tuple[List[sp.Expr], List[sp.Symbol], List[sp.Symbol]]:
        qs = [DoF.symbol() for DoF in DoFs]
        q_dots = [DoF.velocity_symbol() for DoF in DoFs]

        dq_dts = [sp.diff(q(t), t) for q in DoFs]

        def symbolize_expr(expr: sp.Expr, t: sp.Symbol, DoFs: List[DegreeOfFreedom], qs: List[sp.Symbol], q_dots: List[sp.Symbol], dq_dts: List[sp.Expr]) -> sp.Expr:
            for i in range(len(DoFs)):
                # Note: we have to do the derivative substitutions first
                #       if we did the coordinate ones first, the derivatives would turn into derivatives of symbols instead of functions
                #       then the derivative substitutions wouldn't match and would fail

                # Substitute derivative
                expr = expr.subs(dq_dts[i], q_dots[i])
                # Substitute coordinate
                expr = expr.subs(DoFs[i](t), qs[i])
            return expr

        symbolized = [symbolize_expr(expr, t, DoFs, qs, q_dots, dq_dts) for expr in expressions]

        return (symbolized, qs, q_dots)

    def __init__(self, L: sp.Expr, t: sp.Symbol, DoFs: List[DegreeOfFreedom]):
        self._L = L
        self._t = t
        self._DoFs = DoFs

    def __str__(self):
        return str(self.L)
    
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

        # Replace coordinates with the corresponding symbols
        (forces_and_momenta, qs, q_dots) = Lagrangian.symbolize(forces + momenta, t, DoFs)
        forces = forces_and_momenta[0:len(forces)]
        momenta = forces_and_momenta[len(forces):]

        # Generate system of equations to solve for velocities given momenta
        p_qs = [DoF.momentum_symbol() for DoF in DoFs]
        velocity_eqs = []

        for p_q, momentum in zip(p_qs, momenta):
            velocity_eqs.append(sp.Eq(p_q, momentum))

        # Solve the system of equations to get expressions for the velocities
        velocity_solutions, = sp.linsolve(velocity_eqs, q_dots)
        velocities = list(velocity_solutions)

        return Lagrangian.ODEExpressions(t, qs, p_qs, q_dots, forces, momenta, velocities)

class Dissipation:
    """TODO"""
    
    def __init__(self, F: sp.Expr, t: sp.Symbol, DoFs: List[DegreeOfFreedom]):
        self._F = F
        self._t = t
        self._DoFs = DoFs
    
    def solve(self) -> List[sp.Expr]:
        F = self._F
        t = self._t
        DoFs = self._DoFs

        dissipative_forces = []
        for q in DoFs:
            q_dot = sp.diff(q(t), t)
            dF_dqdot = sp.diff(F, q_dot).doit()
            dF_dqdot = sp.simplify(dF_dqdot)
            dissipative_forces.append(dF_dqdot)
        
        (dissipative_forces, _, _) = Lagrangian.symbolize(dissipative_forces, t, DoFs)

        return dissipative_forces

class LagrangianBody:

    class LagrangianPhysics(ABC):
        @abstractmethod
        def DoFs(self) -> List[DegreeOfFreedom]:
            """See LagrangianBody.DoFs"""
            pass
        
        @abstractmethod
        def parameters(self) -> List[sp.Symbol]:
            """See LagrangianBody.parameters"""
            pass

        @abstractmethod
        def U(self, t: sp.Symbol) -> sp.Expr:
            """See LagrangianBody.U"""
            pass
        
        @abstractmethod
        def T(self, t: sp.Symbol) -> sp.Expr:
            """See LagrangianBody.T"""
            pass
        
        @abstractmethod
        def F(self, t: sp.Symbol) -> sp.Expr:
            """See LagrangianBody.F"""
            pass

    def __init__(self, t: sp.Symbol, physics: LagrangianPhysics, *constraints: Constraint):
        self._t = t
        self._physics = physics
        self._constraints = list(constraints)

    @property
    def t(self) -> sp.Symbol: return self._t
    
    @property
    def constraints(self) -> List[Constraint]: return self._constraints

    def DoFs(self) -> List[DegreeOfFreedom]:
        """Returns a list of the coordinates that are degrees of freedom of this body"""
        all_DoFs = self._physics.DoFs()
        unconstrained = Lagrangian.unconstrained_DoFs(all_DoFs, self._constraints)
        return unconstrained
        
    def parameters(self) -> List[sp.Symbol]:
        """Returns a list of the parameters for this body"""
        return self._physics.parameters()

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
        U_free = self._physics.U(t)

        U_constrained = Lagrangian.apply_constraints(t, U_free, self._constraints)
        
        return U_constrained.doit().simplify()
    
    @abstractmethod
    def T(self) -> sp.Expr:
        """Generates and returns a symbolic expression for the kinetic energy of this body

        Arguments:
            `t` : a symbol for the time variable

        Returns:
            A symbolic expression for the kinetic energy of the pendulum
        """
        t = self._t
        T_free = self._physics.T(t)

        T_constrained = Lagrangian.apply_constraints(t, T_free, self._constraints)
        
        return T_constrained.doit().simplify()
    
    @abstractmethod
    def F(self) -> sp.Expr:
        """Generates and returns a symbolic expression for the dissipation function of this body
        
        Arguments:
            `t` : a symbol for the time variable

        Returns:
            A symbolic expression for the dissipation function of this body
        """
        t = self._t
        F_free = self._physics.F(t)

        F_constrained = Lagrangian.apply_constraints(t, F_free, self._constraints)

        return F_constrained.doit().simplify()
    
    def lagrangian(self) -> Lagrangian:
        """Generates and returns the (simplified) Lagrangian for this body"""
        return Lagrangian(self.T() - self.U(), self._t, self.DoFs())
    
    def dissipation(self) -> Dissipation:
        """Generates and returns a Dissipation instance for this body"""
        return Dissipation(self.F(), self._t, self.DoFs())
    
    def constrain(self, *constraints: Constraint):
        return LagrangianBody(self._physics, self._t, constraints)