from abc import ABC, abstractmethod
from typing import List, Tuple, Callable

import numpy as np

from physics.numerical import NumericalBody, TimeEvolver

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


# Class that manages the evolution of the double pendulum over time
class PhysicsSimulation:
    def __init__(self, body: NumericalBody, time_evolver: TimeEvolver, init_state: np.ndarray):#, var_trackers: List[float] = []):
        self._body = body
        self._time_evolver = time_evolver

        self._init_state = init_state
        self._body.state = init_state
        self._elapsed_time = 0

        # self.__t_tracker = VariableTracker(0, lambda t, state, prop, behavior: t)
        # self.__var_trackers = var_trackers

    @property
    def body(self) -> NumericalBody: return self._body
    @property
    def elapsed_time(self) -> float: return self._elapsed_time

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
        self._body.state = self._init_state
        self._elapsed_time = 0

    # Progresses the simulation through a time step, dt
    def step(self, dt: float):
        # Calculate next state
        next_state = self._time_evolver.evolve(self.elapsed_time, self.body.state, dt)
        self._body.state = next_state
        # Update elapsed time
        self._elapsed_time += dt

        # Update variable trackers
        # self.__t_tracker.track(self.__elapsed_time, self.pendulum().state(), self.pendulum().prop(), self.behavior())
        # for tracker in self.__var_trackers:
        #     tracker.track(self.__elapsed_time, self.pendulum().state(), self.pendulum().prop(), self.behavior())

    # Progresses the simulation up to absolute time, t_final, in steps of dt
    def step_until(self, dt: float, t_final: float):
        while (self.elapsed_time < t_final):
            self.step(dt)
    
    # Progresses the simulation through an amount of time, delta_t, in steps of dt
    def step_for(self, dt: float, delta_t: float):
        local_elapsed_time = 0.0
        while (local_elapsed_time < delta_t):
            self.step(dt)
            local_elapsed_time += dt