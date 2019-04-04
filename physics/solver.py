from typing import List, Callable, Tuple, Iterable
from abc import ABC, abstractmethod

import sympy as sp
import numpy as np

from physics.lagrangian import DegreeOfFreedom, Lagrangian, Dissipation, LagrangianBody
from physics.numerical import NumericalODEs, LagrangianNumericalODEs, TimeEvolver, ODEINTSolver, NumericalBody
from physics.simulation import PhysicsSimulation

def lambdify(args: Iterable[sp.Expr], expr: sp.Expr):
    return sp.lambdify(args, expr, modules="numpy")

class Solver:

    def __init__(self, lagrangian_body: LagrangianBody):
        self._lagrangian_body = lagrangian_body
    
    def _numerical_body(t: sp.Symbol, L: Lagrangian, DoFs: List[DegreeOfFreedom], qs: List[sp.Symbol], q_dots: List[sp.Symbol], params: List[sp.Symbol]) -> NumericalBody:
        exprs = Lagrangian.symbolize_exprs([L.U, L.T], t, DoFs)

        U_lambda = lambdify([t] + qs + q_dots + params, exprs[0])
        T_lambda = lambdify([t] + qs + q_dots + params, exprs[1])

        return NumericalBody(U_lambda, T_lambda)
    
    def _time_evolver(t: sp.Symbol, L: Lagrangian, D: Dissipation, DoFs: List[DegreeOfFreedom], qs: List[sp.Symbol], q_dots: List[sp.Symbol], p_qs: List[sp.Symbol], params: List[sp.Symbol]) -> TimeEvolver:
        # Solve the Lagrangian part of the equation of motion
        ode_expr = L.solve()

        force_lambdas    = [lambdify([t] + qs + q_dots + params, force_expr) for force_expr in ode_expr.force_exprs]
        momentum_lambdas = [lambdify([t] + qs + q_dots + params, momentum_expr) for momentum_expr in ode_expr.momentum_exprs]
        velocity_lambdas = [lambdify([t] + qs + p_qs + params, velocity_expr) for velocity_expr in ode_expr.velocity_exprs]

        # Solve the dissipation part of the equation of motion
        dissipative_force_exprs = D.solve()

        dissipative_force_lambdas = [lambdify([t] + qs + q_dots + params, dissipative_force_expr) for dissipative_force_expr in dissipative_force_exprs]

        numerical_odes = LagrangianNumericalODEs(ode_expr.num_q, force_lambdas, momentum_lambdas, velocity_lambdas, dissipative_force_lambdas)
        time_evolver = TimeEvolver(numerical_odes, ODEINTSolver())

        return time_evolver
    
    def simulate(self, init_state: np.ndarray, init_params: np.ndarray) -> PhysicsSimulation:
        # Collect all arguments
        t = self._lagrangian_body.t
        params = self._lagrangian_body.parameters()
        DoFs = self._lagrangian_body.DoFs()
        
        # Generate symbols
        qs = [DoF.symbol for DoF in DoFs]
        q_dots = [DoF.velocity_symbol for DoF in DoFs]
        p_qs = [DoF.momentum_symbol for DoF in DoFs]

        # Calculate expressions
        L = self._lagrangian_body.lagrangian()
        D = self._lagrangian_body.dissipation()

        # Construct solver elements
        body = Solver._numerical_body(t, L, DoFs, qs, q_dots, params)
        time_evolver = Solver._time_evolver(t, L, D, DoFs, qs, q_dots, p_qs, params)
        
        return PhysicsSimulation(body, time_evolver, init_state, init_params)