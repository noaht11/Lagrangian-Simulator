from typing import List, Tuple
from abc import ABC, abstractmethod
from time import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from physics.simulation import PhysicsSimulation

class Artist(ABC):

    @abstractmethod
    def reset(self, axes) -> Tuple:
        pass

    @abstractmethod
    def draw(self, t: float, state: np.ndarray) -> Tuple:
        pass

class PhysicsAnimation:
    def __init__(self, simulation: PhysicsSimulation, artist: Artist):
        self._simulation = simulation
        self._artist = artist

    def init(self):
        """Performs all the setup necessary before running an animation

        This MUST be called before calling run()
        """
        # Creat the figure
        self.fig = plt.figure(figsize=(8, 8))

        # Define how much larger the plot will be than the size of the pendulum
        scale_margin_factor_x = 6
        scale_margin_factor_y = 2
        L = 0.045
        scale_x = (-1 * scale_margin_factor_x * L, scale_margin_factor_x * L)
        scale_y = (-1 * scale_margin_factor_y * L, scale_margin_factor_y * L)

        # Create the subplot
        self.ax_main = self.fig.add_subplot(111, aspect = 'equal', autoscale_on = False, xlim = scale_x, ylim = scale_y)
        self.ax_main.set_axis_off() # Hide the axes

        # Main horizontal axis
        num_points = 50
        self.ax_main.plot(np.linspace(scale_x[0], scale_x[1], num_points), np.zeros((num_points, 1)))

        # Text indicators
        self.time_text_main = self.ax_main.text(0.02, 0.95, '', transform=self.ax_main.transAxes)
        self.energy_text = self.ax_main.text(0.02, 0.85, '', transform=self.ax_main.transAxes)

        self._reset()

    def _reset(self):
        """Resets the animation

        Resets the simulation to its initial conditions
        """
        self._simulation.reset()

        artist_mod = self._artist.reset(self.ax_main) # TODO handle multiple modifications

        self.time_text_main.set_text('')
        self.energy_text.set_text('')

        # Required for matplotlib to update
        return artist_mod + (self.time_text_main, self.energy_text)

    def _animate(self, i: int, dt: float, draw_dt: float):
        """Internal function that performs a single animation step"""

        # Simulate next step
        self._simulation.step_for(dt, draw_dt)

        # Update pendulum position plot
        artist_mod = self._artist.draw(self._simulation.elapsed_time, self._simulation.body.state)

        # Update elapsed time text
        t = self._simulation.elapsed_time
        self.time_text_main.set_text('Time = %.1f s' % t)

        # Update energy text
        potential = self._simulation.body.U(t)
        kinetic = self._simulation.body.T(t)
        total_energy = potential + kinetic
        self.energy_text.set_text('Energy = %7.3f' % (total_energy))

        # Required for matplotlib to update
        # return self.line_main, self.time_text_main, self.energy_text, self.line_var
        return artist_mod + (self.time_text_main, self.energy_text)

    def run(self, dt: float, draw_dt: float, t_final: float):
        """Runs and displays an animation of the pendulum
        
        Arguments:

            dt      : time step for the simulation (seconds)
            draw_dt : time between animation frame updates (seconds)
            t_final : time at which the simulation will stop (seconds)
        
        """
        interval = draw_dt * 1000 # interval is in milliseconds
        frames = int(t_final / dt)

        self.ani = animation.FuncAnimation(self.fig, self._animate, fargs = (dt, draw_dt), frames=frames, interval=interval, blit=True, init_func=self._reset, repeat=False)

        plt.show()