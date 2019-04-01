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
    def state_to_y(self, t: float, state: np.ndarray) -> np.ndarray:
        """TODO"""
        pass
    
    @abstractmethod
    def dy_dt(self, t: float, y: np.ndarray) -> np.ndarray:
        """TODO"""
        pass

    @abstractmethod
    def y_to_state(self, t: float, y: np.ndarray) -> np.ndarray:
        """TODO"""
        pass

class LagrangianNumericalODEs(NumericalODEs):
    """TODO"""

    def __init__(self, num_q: int, forces: List[Callable[..., float]], momenta: List[Callable[..., float]], velocities: List[Callable[..., float]]):
        assert(num_q == len(forces))
        assert(num_q == len(momenta))
        assert(num_q == len(velocities))

        self._num_q = num_q
        self._forces = forces
        self._momenta = momenta
        self._velocities = velocities
    
    def state_to_y(self, t: float, state: np.ndarray) -> np.ndarray:
        """Implementation of superclass method"""
        num_q = self._num_q
        qs = state[0:num_q]
        q_dots = state[num_q:]

        p_qs = np.array([momentum(t, *qs, *q_dots) for momentum in self._momenta])
        # p_qs = list(map(lambda momentum: momentum(t, *qs, *q_dots), self._momenta))

        return np.concatenate((qs, p_qs))
    
    def dy_dt(self, t: float, y: np.ndarray) -> np.ndarray:
        """Implementation of superclass method"""
        num_q = self._num_q
        qs = y[0:num_q]
        p_qs = y[num_q:]

        q_dots = np.array([velocity(t, *qs, *p_qs) for velocity in self._velocities])
        p_q_dots = np.array([force(t, *qs, *q_dots) for force in self._forces])
        # q_dots = list(map(lambda velocity: velocity(t, *qs, *p_qs), self._velocities))
        # p_q_dots = list(map(lambda force: force(t, *qs, *q_dots), self._forces))

        return np.concatenate((q_dots, p_q_dots))

    def y_to_state(self, t: float, y: np.ndarray) -> np.ndarray:
        """Implementation of superclass method"""
        num_q = self._num_q
        qs = y[0:num_q]
        p_qs = y[num_q:]

        q_dots = np.array([velocity(t, *qs, *p_qs) for velocity in self._velocities])
        # q_dots = list(map(lambda velocity: velocity(t, *qs, *p_qs), self._velocities))

        return np.concatenate((qs, q_dots))

class ODESolver(ABC):
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
    def solve_ode(self, t_0: float, y_0: np.ndarray, dy_dt: Callable[[float, np.ndarray], np.ndarray], dt: float) -> np.ndarray:
        pass

import numpy as np
from scipy.integrate import odeint

class ODEINTSolver(ODESolver):
    """ODESolver implementation that uses scipy.integrate.odeint to solve ODEs"""

    def solve_ode(self, t_0: float, y_0: np.ndarray, dy_dt: Callable[[float, np.ndarray], np.ndarray], dt: float) -> np.ndarray:
        return odeint(dy_dt, y_0, np.array([t_0, t_0 + dt]), tfirst = True)[1]

###################################################################################################################################################################################
# TIME EVOLVER
###################################################################################################################################################################################

class TimeEvolver:
    """Class for implementing numerical methods to solve the time evolution"""
    
    def __init__(self, ODEs: NumericalODEs, solver: ODESolver):
        self._ODEs = ODEs
        self._solver = solver

    def evolve(self, t: float, state: np.ndarray, dt: float) -> np.ndarray:
        """Updates the state to it's new state at time t + dt"""
        # Convert the current state to y vector (at time t)
        y_0 = self._ODEs.state_to_y(t, state)

        # Solve the ODE
        y_1 = self._solver.solve_ode(t, y_0, self._ODEs.dy_dt, dt)

        # Convert resulting y vector back to state (at time d + dt now)
        state_1 = self._ODEs.y_to_state(t + dt, y_1)

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
