from abc import ABC, abstractmethod
from typing import List, Tuple, Callable

import numpy as np

from physics.numerical import NumericalBody, TimeEvolver

# Class that manages the evolution of the double pendulum over time
class PhysicsSimulation:
    def __init__(self, body: NumericalBody, time_evolver: TimeEvolver, init_state: np.ndarray, init_params: np.ndarray):
        self._body = body
        self._time_evolver = time_evolver
        self._init_state = init_state
        self._parameters = init_params

        self._body.state = init_state
        self._elapsed_time = 0

    @property
    def body(self) -> NumericalBody: return self._body
    @property
    def elapsed_time(self) -> float: return self._elapsed_time

    @property
    def parameters(self) -> np.ndarray: return self._parameters
    
    @parameters.setter
    def parameters(self, value: np.ndarray): self._parameters = value

    # Resets the pendulum to its initial state and returns to time 0
    def reset(self):
        self._body.state = self._init_state
        self._elapsed_time = 0

    # Progresses the simulation through a time step, dt
    def step(self, dt: float):
        # Calculate next state
        next_state = self._time_evolver.evolve(self.elapsed_time, self.body.state, dt, self.parameters)
        self._body.state = next_state
        # Update elapsed time
        self._elapsed_time += dt

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