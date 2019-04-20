from typing import Callable, List
from abc import ABC, abstractmethod

import sympy as sp
import numpy as np

###################################################################################################################################################################################
# ODE SOLVER CLASSES
###################################################################################################################################################################################

class NumericalODEs(ABC):
    """TODO"""

    @abstractmethod
    def state_to_y(self, t: float, state: np.ndarray, params: np.ndarray) -> np.ndarray:
        """TODO"""
        pass
    
    @abstractmethod
    def dy_dt(self, t: float, y: np.ndarray, params: np.ndarray) -> np.ndarray:
        """TODO"""
        pass

    @abstractmethod
    def y_to_state(self, t: float, y: np.ndarray, params: np.ndarray) -> np.ndarray:
        """TODO"""
        pass

class LagrangianNumericalODEs(NumericalODEs):
    """TODO"""

    def __init__(self, num_q: int, forces: List[Callable[..., float]], momenta: List[Callable[..., float]], velocity_matrix: List[List[Callable[..., float]]], velocity_constant: List[Callable[..., float]], dissipative_forces: List[Callable[..., float]]):
        self._num_q = num_q
        self._forces = forces
        self._momenta = momenta
        self._velocity_matrix = velocity_matrix
        self._velocity_constant = velocity_constant
        self._dissipative_forces = dissipative_forces

    def _velocities(self, t: float, y: np.ndarray, params: np.ndarray) -> np.ndarray:
        num_q = self._num_q
        qs = y[0:num_q]
        p_qs = y[num_q:]

        velocity_constant = np.array([constant(t, *qs, *params) for constant in self._velocity_constant])
        rhs = p_qs - velocity_constant

        velocity_matrix = np.array([[element(t, *qs, *params) for element in row] for row in self._velocity_matrix])
        
        return np.linalg.solve(velocity_matrix, rhs)
    
    def state_to_y(self, t: float, state: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Implementation of superclass method"""
        num_q = self._num_q
        qs = state[0:num_q]
        q_dots = state[num_q:]

        p_qs = np.array([momentum(t, *qs, *q_dots, *params) for momentum in self._momenta])

        return np.concatenate((qs, p_qs))
    
    def dy_dt(self, t: float, y: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Implementation of superclass method"""
        num_q = self._num_q
        qs = y[0:num_q]
        p_qs = y[num_q:]

        q_dots = self._velocities(t, y, params)
        
        forces = np.array([force(t, *qs, *q_dots, *params) for force in self._forces])
        dissipative_forces = np.array([dissipative_force(t, *qs, *q_dots, *params) for dissipative_force in self._dissipative_forces])
        p_q_dots = forces - dissipative_forces

        return np.concatenate((q_dots, p_q_dots))

    def y_to_state(self, t: float, y: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Implementation of superclass method"""
        num_q = self._num_q
        qs = y[0:num_q]
        p_qs = y[num_q:]

        q_dots = self._velocities(t, y, params)

        return np.concatenate((qs, q_dots))

class ODESolver(ABC):
    """Solves an ODE

    The ODE must be defined by:
    
      dy/dt = dy_dt(t, y)
    
    with initial condition:
    
      y(t_0) = y_0
    
    to find:
    
      y(t) between t_0 and t_0 + dt
    """

    @abstractmethod
    def solve_ode(self, t_0: float, y_0: np.ndarray, dy_dt: Callable[[float, np.ndarray], np.ndarray], dt: float, params: np.ndarray) -> np.ndarray:
        pass

import numpy as np
from scipy.integrate import odeint

class ODEINTSolver(ODESolver):
    """ODESolver implementation that uses scipy.integrate.odeint to solve ODEs"""

    def solve_ode(self, t_0: float, y_0: np.ndarray, dy_dt: Callable[[float, np.ndarray], np.ndarray], dt: float, params: np.ndarray) -> np.ndarray:
        return odeint(dy_dt, y_0, np.array([t_0, t_0 + dt]), args = (params,), tfirst = True)[1]

###################################################################################################################################################################################
# TIME EVOLVER
###################################################################################################################################################################################

class TimeEvolver:
    """Class for implementing numerical methods to solve the time evolution"""
    
    def __init__(self, ODEs: NumericalODEs, solver: ODESolver):
        self._ODEs = ODEs
        self._solver = solver

    def evolve(self, t: float, state: np.ndarray, dt: float, params: np.ndarray) -> np.ndarray:
        """Updates the state to it's new state at time t + dt"""
        # Convert the current state to y vector (at time t)
        y_0 = self._ODEs.state_to_y(t, state, params)

        # Solve the ODE
        y_1 = self._solver.solve_ode(t, y_0, self._ODEs.dy_dt, dt, params)

        # Convert resulting y vector back to state (at time d + dt now)
        state_1 = self._ODEs.y_to_state(t + dt, y_1, params)

        # Return updated state
        return state_1

###################################################################################################################################################################################
# NUMERICAL BODY
###################################################################################################################################################################################
class NumericalBody:

    def __init__(self, U: Callable[[float], float], T: Callable[[float], float]):
        self._U = U
        self._T = T

    @property
    def state(self) -> np.ndarray:
        return self._state
    
    @state.setter
    def state(self, value: np.ndarray):
        self._state = value
    
    def U(self, t: float) -> float:
        return self._U(t, *self._state)
    
    def T(self, t: float) -> float:
        return self._T(t, *self._state)