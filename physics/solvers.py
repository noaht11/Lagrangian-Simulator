from typing import Callable, List
from abc import ABC, abstractmethod

import sympy as sp

from physics.lagrangian import Lagrangian, Constraint

class NumericalSolver(ABC):
    """TODO"""

    @abstractmethod
    def state_to_y(self, t: float, state: List[float]) -> List[float]:
        """TODO"""
        pass
    
    @abstractmethod
    def dy_dt(self, t: float, y: List[float]) -> List[float]:
        """TODO"""
        pass

    @abstractmethod
    def y_to_state(self, t: float, y: List[float]) -> List[float]:
        """TODO"""
        pass

class LagrangianNumericalSolver(NumericalSolver):
    """TODO"""

    def from_ode_expr(odeExpr: Lagrangian.ODEExpressions):
        """TODO"""
        return LagrangianNumericalSolver(odeExpr.num_q, odeExpr.force_lambdas(), odeExpr.momentum_lambdas(), odeExpr.velocity_lambdas())

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