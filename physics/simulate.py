from abc import ABC, abstractmethod
from typing import List, Tuple, Callable

from physics.solvers import NumericSolver

###################################################################################################################################################################################
# ABSTRACT BASE CLASSES
###################################################################################################################################################################################

# Abstract Base Class for implementing numerical methods to solve the time evolution
class TimeEvolver(ABC):

    def evolve(self, t: float, state: List[float], solver: NumericSolver, dt: float) -> List[float]:
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

###################################################################################################################################################################################
# TIME EVOLVER IMPLEMENTATIONS
###################################################################################################################################################################################

import numpy as np
from scipy.integrate import odeint

# TimeEvolver implementation that uses scipy.integrate.odeint to solve ODEs
class ODEINTTimeEvolver(TimeEvolver):
    def solve_ode(self, t_0: float, y_0: List[float], dy_dt: Callable[[float, List[float]], List[float]], dt: float) -> List[float]:
        return odeint(dy_dt, y_0, [t_0, t_0 + dt], tfirst = True)[1]

###################################################################################################################################################################################
# SIMULATION
###################################################################################################################################################################################

# class VariableTracker():

#     def __init__(self, init_val: float, tracker: Callable[[float, DoublePendulum.State, DoublePendulum.Properties, DoublePendulumBehavior], float]):
#         self.data = np.array([init_val])
#         self.__tracker = tracker
        
#     def track(self, t: float, state: DoublePendulum.State, prop: DoublePendulum.Properties, behavior: DoublePendulumBehavior):
#         new_val = self.__tracker(t, state, prop, behavior)
#         self.data = np.append(self.data, new_val)
    
#     def min(self):
#         return np.min(self.data)
    
#     def max(self):
#         return np.max(self.data)

class SimulationBody(ABC):

    @property
    @abstractmethod
    def state(self):
        pass
    
    @state.setter
    @abstractmethod
    def state(self, value):
        pass


# Class that manages the evolution of the double pendulum over time
class Simulation:
    def __init__(self, body: SimulationBody, time_evolver: TimeEvolver):#, var_trackers: List[float] = []):
        self._body = body
        self._time_evolver = time_evolver

        self._init_state = self._body.state
        self._elapsed_time = 0

        # self.__t_tracker = VariableTracker(0, lambda t, state, prop, behavior: t)
        # self.__var_trackers = var_trackers

    def body(self) -> SimulationBody: return self._body
    def time_evolver(self) -> TimeEvolver: return self.__time_evolver
    
    def elapsed_time(self) -> float: return self.__elapsed_time

    # def t_tracker(self) -> VariableTracker: return self.__t_tracker
    # def var_trackers(self) -> List[VariableTracker]: return self.__var_trackers

    # Calculates the current energy (potential and kinetic) of the pendulum
    #
    # Return value is a tuple of the form: (potential, kinetic)
    #
    # def energy(self) -> Tuple[float, float]:
    #     potential = self.__behavior.energy_potential(self.__elapsed_time, self.__pendulum.state(), self.__pendulum.prop())
    #     kinetic = self.__pendulum.energy_kinetic()
    #     return (potential, kinetic)

    # Resets the pendulum to its initial state and returns to time 0
    def reset(self):
        self._body.state = self.init_state
        self._elapsed_time = 0

    # Progresses the simulation through a time step, dt
    def step(self, dt: float):
        # Calculate next state
        self.time_evolver().evolve(self.pendulum(), self.behavior(), self.__elapsed_time, dt)
        # Update elapsed time
        self.__elapsed_time += dt

        # Update variable trackers
        self.__t_tracker.track(self.__elapsed_time, self.pendulum().state(), self.pendulum().prop(), self.behavior())
        for tracker in self.__var_trackers:
            tracker.track(self.__elapsed_time, self.pendulum().state(), self.pendulum().prop(), self.behavior())

    # Progresses the simulation up to absolute time, t_final, in steps of dt
    def step_until(self, dt: float, t_final: float):
        while (self.elapsed_time() < t_final):
            self.step(dt)
    
    # Progresses the simulation through an amount of time, delta_t, in steps of dt
    def step_for(self, dt: float, delta_t: float):
        local_elapsed_time = 0.0
        while (local_elapsed_time < delta_t):
            self.step(dt)
            local_elapsed_time += dt