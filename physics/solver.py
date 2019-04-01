from typing import List, Callable, Tuple
from abc import ABC, abstractmethod

import sympy as sp

from physics.lagrangian import Lagrangian, LagrangianBody
from physics.numerical import NumericalODEs, TimeEvolver, ODEINTSolver, NumericalBody
from physics.simulation import PhysicsSimulation

class LagrangianNumericalODEs(NumericalODEs):
    """TODO"""

    def from_ode_expr(ode_expr: Lagrangian.ODEExpressions):
        """TODO"""
        return LagrangianNumericalODEs(ode_expr.num_q, ode_expr.force_lambdas(), ode_expr.momentum_lambdas(), ode_expr.velocity_lambdas())

    def __init__(self, num_q: int, forces: List[Callable], momenta: List[Callable], velocities: List[Callable]):
        assert(num_q == len(forces))
        assert(num_q == len(momenta))
        assert(num_q == len(velocities))

        self._num_q = num_q
        self._forces = forces
        self._momenta = momenta
        self._velocities = velocities
    
    def state_to_y(self, t: float, state: List[float]) -> List[float]:
        """Implementation of superclass method"""
        num_q = self._num_q
        qs = state[0:num_q]
        q_dots = state[num_q:]

        p_qs = list(map(lambda momentum: momentum(t, *qs, *q_dots), self._momenta))

        return qs + p_qs
    
    def dy_dt(self, t: float, y: List[float]) -> List[float]:
        """Implementation of superclass method"""
        num_q = self._num_q
        qs = y[0:num_q]
        p_qs = y[num_q:]

        q_dots = list(map(lambda velocity: velocity(t, *qs, *p_qs), self._velocities))
        p_q_dots = list(map(lambda force: force(t, *qs, *q_dots), self._forces))

        return q_dots + p_q_dots

    def y_to_state(self, t: float, y: List[float]) -> List[float]:
        """Implementation of superclass method"""
        num_q = self._num_q
        qs = y[0:num_q]
        p_qs = y[num_q:]

        q_dots = list(map(lambda velocity: velocity(t, *qs, *p_qs), self._velocities))

        return qs + q_dots

class Solver:

    def __init__(self, lagrangian_body: LagrangianBody):
        self._lagrangian_body = lagrangian_body
    
    def _numerical_body(self) -> NumericalBody:
        t = self._lagrangian_body.t
        DoFs = self._lagrangian_body.DoFs()

        U_expr = self._lagrangian_body.U()
        T_expr = self._lagrangian_body.T()

        (exprs, qs, q_dots) = Lagrangian.symbolize([U_expr, T_expr], t, DoFs)

        U_lambda = sp.lambdify([t] + qs + q_dots, exprs[0])
        T_lambda = sp.lambdify([t] + qs + q_dots, exprs[1])

        return NumericalBody(U_lambda, T_lambda)
    
    def _time_evolver(self) -> TimeEvolver:
        ode_expr = self._lagrangian_body.lagrangian().solve()
        numerical_odes = LagrangianNumericalODEs.from_ode_expr(ode_expr)
        time_evolver = TimeEvolver(numerical_odes, ODEINTSolver())

        return time_evolver
    
    def simulate(self, init_state: List[float]) -> PhysicsSimulation:
        body = self._numerical_body()
        time_evolver = self._time_evolver()

        return PhysicsSimulation(body, time_evolver, init_state)