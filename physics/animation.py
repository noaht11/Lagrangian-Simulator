from typing import List, Tuple
from abc import ABC, abstractmethod
from time import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button

from physics.simulation import PhysicsSimulation

class Artist(ABC):

    @abstractmethod
    def init(self, axes):
        pass

    @abstractmethod
    def draw(self, t: float, state: np.ndarray, params: np.ndarray) -> Tuple:
        pass

class PhysicsAnimation:

    class AnimationConfig:

        def __init__(self, size: float = 1.0, en_start_stop: bool = False):
            self.size = size
            self.en_start_stop = en_start_stop

    def __init__(self, simulation: PhysicsSimulation, artist: Artist, config: AnimationConfig):
        self._simulation = simulation
        self._artist = artist
        self._config = config

        self._started = True

    def _toggle_start(self, event):
        self._started = not self._started

        self._update_start_stop_text()

    def _update_start_stop_text(self):
        if (self._started):
            self._btn_start_stop.label.set_text("Stop")
        else:
            self._btn_start_stop.label.set_text("Run")
        plt.draw()

    def init(self):
        """Performs all the setup necessary before running an animation

        This MUST be called before calling run()
        """
        # Creat the figure
        self.fig = plt.figure(figsize=(8, 8))

        # Define the bounds of the plot
        size = self._config.size
        scale_x = (-size, size)
        scale_y = (-size, size)

        # Main subplot
        self.ax_main = self.fig.add_subplot(111, aspect = "equal", autoscale_on = False, xlim = scale_x, ylim = scale_y)
        self.ax_main.set_axis_off() # Hide the axis lines

        # Initialize artist
        self._artist.init(self.ax_main)

        # Main horizontal axis
        num_points = 50
        self.ax_main.plot(np.linspace(scale_x[0], scale_x[1], num_points), np.zeros((num_points, 1)))

        # Text indicators
        self.time_text_main = self.ax_main.text(0.02, 0.95, "", transform=self.ax_main.transAxes)

        # Start / stop button
        if (self._config.en_start_stop):
            self._started = False

            ax_start_stop = plt.axes([0.8, 0.025, 0.1, 0.04])
            self._btn_start_stop = Button(ax_start_stop, "", color = "lightgoldenrodyellow", hovercolor="0.975")
            self._update_start_stop_text()
            self._btn_start_stop.on_clicked(self._toggle_start)

    def _reset(self):
        """Resets the animation

        Resets the simulation to its initial conditions
        """
        # Reset simulation
        self._simulation.reset()

        # Reset artist
        artist_mod = self._artist.draw(self._simulation.elapsed_time, self._simulation.body.state, self._simulation.parameters)

        # Reset time text
        self.time_text_main.set_text('')

        # Return modified items
        return artist_mod + (self.time_text_main,)

    def _animate(self, i: int, dt: float, draw_dt: float):
        """Internal function that performs a single animation step"""

        # Simulate next step
        
        if (self._started is True):
            self._simulation.step_for(dt, draw_dt)

        # Update pendulum position plot
        artist_mod = self._artist.draw(self._simulation.elapsed_time, self._simulation.body.state, self._simulation.parameters)

        # Update elapsed time text
        t = self._simulation.elapsed_time
        self.time_text_main.set_text('Time = %.1f s' % t)

        # Return modified items
        return artist_mod + (self.time_text_main,)

    def run(self, dt: float, draw_dt: float, t_final: float = -1):
        """Runs and displays an animation of the pendulum
        
        Arguments:

            dt      : time step for the simulation (seconds)
            draw_dt : time between animation frame updates (seconds)
            t_final : time at which the simulation will stop (seconds)
                      if -1 or if en_start_stop was True, the animation will run indefinitely
        
        """
        interval = draw_dt * 1000 # interval is in milliseconds

        frames = None
        if (t_final != -1 and self._config.en_start_stop is not True):
            frames = int(t_final / dt)

        self.ani = animation.FuncAnimation(self.fig, self._animate, fargs = (dt, draw_dt), frames=frames, interval=interval, blit=True, init_func=self._reset, repeat=False)

        plt.show()