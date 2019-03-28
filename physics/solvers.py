from typing import Callable, List
from abc import ABC, abstractmethod

from physics.lagrangian import Lagrangian, Constraint

import sympy as sp

class NumericSolver(ABC):

    @abstractmethod
    def state_to_y(self, t: float, state: List[float]) -> List[float]:
        pass
    
    @abstractmethod
    def dy_dt(self, t: float, y: List[float]) -> List[float]:
        pass

    @abstractmethod
    def y_to_state(self, t: float, y: List[float]) -> List[float]:
        pass

class FunctionNumericSolver(NumericSolver):
    def __init__(self,
        state_to_y: Callable[[float, List[float]], List[float]],
        dy_dt: Callable[[float, List[float]], List[float]],
        y_to_state: Callable[[float, List[float]], List[float]]):

        self._state_to_y = state_to_y
        self._dy_dt = dy_dt
        self._y_to_state = y_to_state
    
    def state_to_y(self, t: float, state: List[float]) -> List[float]:
        return self._state_to_y(t, state)
    
    def dy_dt(self, t: float, y: List[float]) -> List[float]:
        return self._dy_dt(t, y)

    def y_to_state(self, t: float, y: List[float]) -> List[float]:
        return self._y_to_state(t, y)