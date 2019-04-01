from typing import List
from abc import ABC, abstractmethod
from time import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from physics.simulation import PhysicsSimulation

class Artist(ABC):

    @abstractmethod
    def reset(self, axes):
        pass

    @abstractmethod
    def draw(self, state: np.ndarray):
        pass

class PhysicsAnimation:
    def __init__(self, simulation: PhysicsSimulation, artist: Artist):#, t_init_range: Tuple[float, float] = (0, 10), var_init_range: Tuple[float, float] = (-1, 1)):
        self._simulation = simulation
        self._artist = artist
        # self.__t_init_range = t_init_range
        # self.__var_init_range = var_init_range

    # Performs all the setup necessary before running an animation
    #
    # This MUST be called before calling run()
    #
    def init(self):
        # Creat the figure
        self.fig = plt.figure(figsize=(8, 8))

        # Define how much larger the plot will be than the size of the pendulum
        scale_margin_factor_x = 6
        scale_margin_factor_y = 2
        L = 5#self._simulation.pendulum().prop().L()
        scale_x = (-1 * scale_margin_factor_x * L, scale_margin_factor_x * L)
        scale_y = (-1 * scale_margin_factor_y * L, scale_margin_factor_y * L)

        # Create the subplot
        self.ax_main = self.fig.add_subplot(211, aspect = 'equal', autoscale_on = False, xlim = scale_x, ylim = scale_y)
        self.ax_main.set_axis_off() # Hide the axes

        # Main horizontal axis
        num_points = 50
        self.ax_main.plot(np.linspace(scale_x[0], scale_x[1], num_points), np.zeros((num_points, 1)))

        # Text indicators
        self.time_text_main = self.ax_main.text(0.02, 0.95, '', transform=self.ax_main.transAxes)
        self.energy_text = self.ax_main.text(0.02, 0.85, '', transform=self.ax_main.transAxes)

        # Graph figure
        #self.fig_graph = plt.figure(figsize=(8, 8))
        # self.ax_var = self.fig.add_subplot(212, autoscale_on = True, xlim = self.__t_init_range, ylim = self.__var_init_range)
        # self.ax_var.set_xlabel("Time (seconds)")
        # self.ax_var.set_ylabel("Variable")
        # self.ax_var.grid()
        # self.line_var, = self.ax_var.plot([], [])

        self._reset()

    # Resets the simulation to its initial conditions
    # Resets all data and labels to default values
    def _reset(self):
        self._simulation.reset()

        artist_mod = self._artist.reset(self.ax_main) # TODO handle multiple modifications

        self.time_text_main.set_text('')
        self.energy_text.set_text('')

        # self.line_var.set_data([], [])

        # Required for matplotlib to update
        # return artist_mod, self.time_text_main, self.energy_text, self.line_var
        return tuple([*artist_mod, self.time_text_main, self.energy_text])

    # def _relim(self, ax_min, ax_max, var_tracker: VariableTracker) -> Tuple[float, float, bool]:
    #     var_min = var_tracker.min()
    #     var_max = var_tracker.max()
    #     var_mean = (var_min + var_max) / 2

    #     changed = False

    #     # Check min
    #     if (var_tracker.min() < ax_min):
    #         ax_min = var_mean - ((var_mean - var_min) * 2)
    #         changed = True
    #     # Check max
    #     if (var_tracker.max() > ax_max):
    #         ax_max = var_mean + ((var_max - var_mean) * 2)
    #         changed = True
        
    #     return (ax_min, ax_max, changed)

    # Internal function that performs a single animation step
    def _animate(self, i: int, dt: float, draw_dt: float):
        # Simulate next step
        self._simulation.step_for(dt, draw_dt)

        # Update pendulum position plot
        artist_mod = self._artist.draw(self._simulation.body.state)

        # Update elapsed time text
        t = self._simulation.elapsed_time
        self.time_text_main.set_text('Time = %.1f s' % t)

        # Update energy text
        potential = self._simulation.body.U(t)
        kinetic = self._simulation.body.T(t)
        total_energy = potential + kinetic
        # self.energy_text.set_text('Potential = %7.3f\nKinetic = %7.3f\nTotal Energy = %7.3f' % (potential, kinetic, total_energy))
        self.energy_text.set_text('Energy = %7.3f' % (total_energy))

        # Update variable tracker plot
        # t_tracker = self._simulation.t_tracker()
        # var_tracker = self._simulation.var_trackers()[0]
        # self.line_var.set_data(t_tracker.data, var_tracker.data) # TODO For now we only plot the first variable tracker
        
        # Possibly change axis limits
        # (cur_x_min, cur_x_max) = self.ax_var.get_xlim()
        # (cur_y_min, cur_y_max) = self.ax_var.get_ylim()
        # (x_min, x_max, x_changed) = self.__relim(cur_x_min, cur_x_max, t_tracker)
        # (y_min, y_max, y_changed) = self.__relim(cur_y_min, cur_y_max, var_tracker)
        # self.ax_var.set_xlim((x_min, x_max))
        # self.ax_var.set_ylim((y_min, y_max))
        # if (x_changed or y_changed):
        #     plt.draw()

        # Required for matplotlib to update
        # return self.line_main, self.time_text_main, self.energy_text, self.line_var
        return [*artist_mod, self.time_text_main, self.energy_text]

    # Runs and displays an animation of the pendulum
    #
    #   dt       = time step for the simulation (seconds)
    #   draw_dt  = time between animation frame updates (seconds)
    #   t_final  = time at which the simulation will stop (seconds)
    #
    def run(self, dt: float, draw_dt: float, t_final: float):
        interval = draw_dt * 1000 # interval is in milliseconds
        frames = int(t_final / dt)

        self.ani = animation.FuncAnimation(self.fig, self._animate, fargs = (dt, draw_dt), frames=frames, interval=interval, blit=True, init_func=self._reset, repeat=False)

        plt.show()