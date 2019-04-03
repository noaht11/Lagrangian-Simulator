from typing import List, Callable, Tuple, Iterable
from abc import ABC, abstractmethod

import sympy as sp
import numpy as np

from physics.lagrangian import Lagrangian, LagrangianBody
from physics.numerical import NumericalODEs, LagrangianNumericalODEs, TimeEvolver, ODEINTSolver, NumericalBody
from physics.simulation import PhysicsSimulation

def lambdify(args: Iterable[sp.Expr], expr: sp.Expr):
    return sp.lambdify(args, expr, modules="numpy")

class Solver:

    def __init__(self, lagrangian_body: LagrangianBody):
        self._lagrangian_body = lagrangian_body
    
    def _numerical_body(self) -> NumericalBody:
        t = self._lagrangian_body.t
        params = self._lagrangian_body.parameters()
        DoFs = self._lagrangian_body.DoFs()
        
        qs = [DoF.symbol for DoF in DoFs]
        q_dots = [DoF.velocity_symbol for DoF in DoFs]

        U_expr = self._lagrangian_body.U()
        T_expr = self._lagrangian_body.T()
        exprs = Lagrangian.symbolize_exprs([U_expr, T_expr], t, DoFs)

        U_lambda = lambdify([t] + qs + q_dots + params, exprs[0])
        T_lambda = lambdify([t] + qs + q_dots + params, exprs[1])

        return NumericalBody(U_lambda, T_lambda)
    
    def _time_evolver(self) -> TimeEvolver:
        t = self._lagrangian_body.t
        params = self._lagrangian_body.parameters()
        DoFs = self._lagrangian_body.DoFs()
        
        qs = [DoF.symbol for DoF in DoFs]
        q_dots = [DoF.velocity_symbol for DoF in DoFs]
        p_qs = [DoF.momentum_symbol for DoF in DoFs]

        # Solve the Lagrangian part of the equation of motion
        ode_expr = self._lagrangian_body.lagrangian().solve()

        force_lambdas    = [lambdify([t] + qs + q_dots + params, force_expr) for force_expr in ode_expr.force_exprs]
        momentum_lambdas = [lambdify([t] + qs + q_dots + params, momentum_expr) for momentum_expr in ode_expr.momentum_exprs]
        velocity_lambdas = [lambdify([t] + qs + p_qs + params, velocity_expr) for velocity_expr in ode_expr.velocity_exprs]

        # Solve the dissipation part of the equation of motion
        dissipative_force_exprs = self._lagrangian_body.dissipation().solve()

        dissipative_force_lambdas = [lambdify([t] + qs + q_dots + params, dissipative_force_expr) for dissipative_force_expr in dissipative_force_exprs]

        numerical_odes = LagrangianNumericalODEs(ode_expr.num_q, force_lambdas, momentum_lambdas, velocity_lambdas, dissipative_force_lambdas)
        time_evolver = TimeEvolver(numerical_odes, ODEINTSolver())

        return time_evolver
    
    def simulate(self, init_state: np.ndarray, init_params: np.ndarray) -> PhysicsSimulation:
        body = self._numerical_body()
        time_evolver = self._time_evolver()
        
        return PhysicsSimulation(body, time_evolver, init_state, init_params)