from abc import ABC, abstractmethod
from typing import List, Tuple, Callable

from pendulum.core import *
from pendulum.behavior import *

###################################################################################################################################################################################
# ABSTRACT BASE CLASSES
###################################################################################################################################################################################

# Abstract Base Class for implementing numerical methods to solve the time evolution of a Double Pendulum
class TimeEvolver(ABC):

    # Updates the state of the pendulum to it's new state at time t + dt according to the provided behavior
    def evolve(self, pendulum : DoublePendulum, behavior: DoublePendulumBehavior, t: float, dt: float):
        # Convert the current state to y vector (at time t)
        state_0 = pendulum.state()
        y_0 = behavior.state_to_y(t, state_0, pendulum.prop())

        # Solve the ODE
        y_1 = self.solve_ode(t, dt, behavior.dy_dt, y_0, pendulum.prop())

        # Convert resulting y vector back to state (at time d + dt now)
        state_1 = behavior.y_to_state(t + dt, y_1, pendulum.prop())

        # Update the pendulum
        pendulum.set_state(state_1)
    
    # Solves the ODE defined by:
    #
    #   dy/dt = dy_dt(y)
    # 
    # with initial condition:
    #
    #   y(t_0) = y_0
    #
    # to find:
    #
    #   y(t) between t_0 and t_0 + dt
    #
    # NOTE: the absolute time t_0 is not accurate nor meaningful here so dy/dt should not depend on t_0
    @abstractmethod
    def solve_ode(self, t_0: float, dt: float, dy_dt: Callable[[float, List[float], DoublePendulum.Properties], List[float]], y_0: List[float], prop: DoublePendulum.Properties):
        pass

###################################################################################################################################################################################
# TIME EVOLVER IMPLEMENTATIONS
###################################################################################################################################################################################

import numpy as np
from scipy.integrate import odeint

# TimeEvolver implementation that uses scipy.integrate.odeint to solve ODEs
class ODEINTTimeEvolver(TimeEvolver):
    def solve_ode(self, t_0: float, dt: float, dy_dt: Callable[[float, List[float], DoublePendulum.Properties], List[float]], y_0: List[float], prop: DoublePendulum.Properties):
        return odeint(dy_dt, y_0, [t_0, t_0 + dt], args = (prop,), tfirst = True)[1]

###################################################################################################################################################################################
# SIMULATION
###################################################################################################################################################################################

# Class that manages the evolution of the double pendulum over time
class DoublePendulumSimulation:
    def __init__(self, pendulum: DoublePendulum, behavior: DoublePendulumBehavior, time_evolver: TimeEvolver):
        self.__pendulum = pendulum
        self.__behavior = behavior
        self.__time_evolver = time_evolver

        self.init_state = self.__pendulum.state()
        self.__elapsed_time = 0

        self.__t = np.array(0)
        self.__q = np.array(pendulum.state().q())

    def pendulum(self) -> DoublePendulum: return self.__pendulum
    def behavior(self) -> DoublePendulumBehavior: return self.__behavior
    def time_evolver(self) -> TimeEvolver: return self.__time_evolver
    
    def elapsed_time(self) -> float: return self.__elapsed_time

    def q(self) -> List[float]: return self.__q
    def t(self) -> List[float]: return self.__t

    # Calculates the current energy (potential and kinetic) of the pendulum
    #
    # Return value is a tuple of the form: (potential, kinetic)
    #
    def energy(self) -> Tuple[float, float]:
        potential = self.__behavior.energy_potential(self.__elapsed_time, self.__pendulum.state(), self.__pendulum.prop())
        kinetic = self.__pendulum.energy_kinetic()
        return (potential, kinetic)

    # Resets the pendulum to its initial state and returns to time 0
    def reset(self):
        self.__pendulum.set_state(self.init_state)
        self.__elapsed_time = 0

    # Progresses the simulation through a time step, dt
    def step(self, dt: float):
        # Calculate next state
        self.time_evolver().evolve(self.pendulum(), self.behavior(), self.__elapsed_time, dt)
        # Update elapsed time
        self.__elapsed_time += dt

        # Update graph arrays
        self.__t = np.append(self.__t, self.__elapsed_time)
        self.__q = np.append(self.__q, self.pendulum().state().q())

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