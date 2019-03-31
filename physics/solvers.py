from typing import Callable, List
from abc import ABC, abstractmethod

import sympy as sp

###################################################################################################################################################################################
# NUMERICAL SOLVERS
###################################################################################################################################################################################

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

###################################################################################################################################################################################
# TIME EVOLVERS
###################################################################################################################################################################################

class TimeEvolver(ABC):
    """Abstract Base Class for implementing numerical methods to solve the time evolution"""

    def evolve(self, t: float, state: List[float], solver: NumericalSolver, dt: float) -> List[float]:
        """Updates the state to it's new state at time t + dt according to the provided solver"""
        # Convert the current state to y vector (at time t)
        y_0 = solver.state_to_y(t, state)

        # Solve the ODE
        y_1 = self.solve_ode(t, y_0, solver.dy_dt, dt)

        # Convert resulting y vector back to state (at time d + dt now)
        state_1 = solver.y_to_state(t + dt, y_1)

        # Return updated state
        return state_1
    
    # Solves the ODE defined by:
    #
    #   dy/dt = dy_dt(t, y)
    # 
    # with initial condition:
    #
    #   y(t_0) = y_0
    #
    # to find:
    #
    #   y(t) between t_0 and t_0 + dt
    #
    @abstractmethod
    def solve_ode(self, t_0: float, y_0: List[float], dy_dt: Callable[[float, List[float]], List[float]], dt: float) -> List[float]:
        pass

import numpy as np
from scipy.integrate import odeint

# TimeEvolver implementation that uses scipy.integrate.odeint to solve ODEs
class ODEINTTimeEvolver(TimeEvolver):
    def solve_ode(self, t_0: float, y_0: List[float], dy_dt: Callable[[float, List[float]], List[float]], dt: float) -> List[float]:
        return odeint(dy_dt, y_0, [t_0, t_0 + dt], tfirst = True)[1]

###################################################################################################################################################################################
# NUMERICAL BODY
###################################################################################################################################################################################
class NumericalBody:

    def __init__(self, U: Callable[[float], float], T: Callable[[float], float]):
        self._U = U
        self._T = T

    @property
    def state(self) -> List[float]:
        return self._state
    
    @state.setter
    def state(self, value: List[float]):
        self._state = value
    
    def U(self, t: float) -> float:
        return self._U(t)
    
    def T(self, t: float) -> float:
        return self._T(t)
