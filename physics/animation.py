from typing import List, Tuple, Callable
from abc import ABC, abstractmethod
import os
from platform import system
from pathlib import Path
from time import time

import numpy as np
import matplotlib as mpl
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

    # Toolbar configuration
    mpl.rcParams["toolbar"] = "None"

    # ImageMagick Configuration
    """Path to modules folder of ImageMagick"""
    MAGICK_ENV = "MAGICK_HOME"

    """Name of ImageMagick executable"""
    MAGICK_EXE = "magick.exe"

    MAGICK_ENABLED = MAGICK_ENV in os.environ
    if (MAGICK_ENABLED):
        mpl.rcParams["animation.convert_path"] = str((Path(os.environ[MAGICK_ENV]).parent.parent / MAGICK_EXE).resolve())

    class AnimationConfig:

        MODE_AUTONOMOUS = 0
        MODE_INTERACTIVE = 1

        def __init__(self, size: float = 1.0, mode: int = MODE_AUTONOMOUS, en_reset: bool = False, en_start_stop: bool = False, save_gif_path: str = None, save_fps: int = 30):
            self.size = size
            self.mode = mode
            self.en_reset = en_reset
            self.en_start_stop = en_start_stop
            self.save_gif_path = save_gif_path
            self.save_fps = save_fps

    class Parameter:

        def __init__(self, name: str, min_val: float, max_val: float, step: float, init_val: float, on_value_changed: Callable[[np.ndarray, float], np.ndarray]):
            self.name = name
            self.min_val = min_val
            self.max_val = max_val
            self.step = step
            self.init_val = init_val
            self.on_value_changed = on_value_changed

    def __init__(self, simulation: PhysicsSimulation, artist: Artist, config: AnimationConfig, parameters: List[Parameter] = []):
        self._simulation = simulation
        self._artist = artist
        self._config = config
        self._parameters = parameters

        self._started = True

    def _reset_clicked(self, event):
        self._reset()

    def _toggle_start(self, event):
        self._started = not self._started

        self._update_start_stop_text()

    def _update_start_stop_text(self):
        if (self._started):
            self._btn_start_stop.label.set_text("Pause")
        else:
            self._btn_start_stop.label.set_text("Play")
        self.fig.canvas.draw_idle()

    def init(self):
        """Performs all the setup necessary before running an animation

        This MUST be called before calling run()
        """
        # Create the figure
        self.fig = plt.figure(figsize=(9, 9))

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

        # Interactive
        if (self._config.mode == PhysicsAnimation.AnimationConfig.MODE_INTERACTIVE):
            self._init_interactive()

    def _init_interactive(self):
        # Make room for the sliders etc.
        self.fig.subplots_adjust(bottom=0.25)

        btn_color = "lightgoldenrodyellow"
        btn_color_hover = "0.975"
        btn_width = 0.1
        btn_height = 0.04
        btn_left = 0.85

        btn_first_bottom = 0.025

        # Reset button
        if (self._config.en_reset):
            ax_reset = plt.axes([btn_left, btn_first_bottom, btn_width, btn_height])
            self._btn_reset = Button(ax_reset, "Reset", color = btn_color, hovercolor = btn_color_hover)
            self._btn_reset.on_clicked(self._reset_clicked)

        # Start / stop button
        if (self._config.en_start_stop):
            self._started = False

            ax_start_stop = plt.axes([btn_left, btn_first_bottom + btn_height, btn_width, btn_height])
            self._btn_start_stop = Button(ax_start_stop, "", color = btn_color, hovercolor = btn_color_hover)
            self._update_start_stop_text()
            self._btn_start_stop.on_clicked(self._toggle_start)

        # Parameters
        sld_color = "lightgoldenrodyellow"
        sld_width = 0.6
        sld_height = 0.02
        sld_left = 0.15
        sld_gap = 0.01

        sld_bottom = sld_gap * 2

        self._sliders = []
        for parameter in self._parameters:
            ax_param = plt.axes([sld_left, sld_bottom, sld_width, sld_height], facecolor = sld_color)
            sld_param = Slider(ax_param, parameter.name, parameter.min_val, parameter.max_val, valinit = parameter.init_val, valstep = parameter.step)
            sld_param.on_changed(parameter.on_value_changed)
            self._sliders.append(sld_param)
            
            sld_bottom += sld_height + sld_gap

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
        if (t_final != -1 and (self._config.mode == PhysicsAnimation.AnimationConfig.MODE_AUTONOMOUS or self._config.en_start_stop is not True)):
            frames = int(t_final / dt)

        self.ani = animation.FuncAnimation(self.fig, self._animate, fargs = (dt, draw_dt), frames=frames, interval=interval, blit=True, init_func=self._reset, repeat=False)

        if(self._config.mode == PhysicsAnimation.AnimationConfig.MODE_AUTONOMOUS):
            if (self._config.save_gif_path is not None):
                writer = animation.ImageMagickWriter(fps = self._config.save_fps)
                self.ani.save(self._config.save_gif_path, writer = writer)
        
        plt.show()