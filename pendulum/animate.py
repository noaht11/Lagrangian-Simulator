from typing import Tuple

from time import time

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from pendulum.core import *
from pendulum.simulate import *

###################################################################################################################################################################################
# ANIMATORS
###################################################################################################################################################################################

# Animates a DoublePendulumSimulation
class DoublePendulumAnimator:
    def __init__(self, simulation: DoublePendulumSimulation):
        self.__simulation = simulation

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
        L = self.__simulation.pendulum().prop().L()
        scale_x = (-1 * scale_margin_factor_x * L, scale_margin_factor_x * L)
        scale_y = (-1 * scale_margin_factor_y * L, scale_margin_factor_y * L)

        # Create the subplot
        self.ax_main = self.fig.add_subplot(211, aspect = 'equal', autoscale_on = False, xlim = scale_x, ylim = scale_y)
        self.ax_main.set_axis_off() # Hide the axes

        # The plot that will show the lines of the pendulum
        self.line_main, = self.ax_main.plot([], [], '-', lw=4)

        # A horizontal line at the pivot point of the pendulum
        num_points = 50
        self.ax_main.plot(np.linspace(scale_x[0], scale_x[1], num_points), np.zeros((num_points, 1)))

        # Text indicators
        self.time_text_main = self.ax_main.text(0.02, 0.95, '', transform=self.ax_main.transAxes)
        self.energy_text = self.ax_main.text(0.02, 0.85, '', transform=self.ax_main.transAxes)

        # Graph figure
        #self.fig_graph = plt.figure(figsize=(8, 8))
        self.ax_var = self.fig.add_subplot(212, autoscale_on = True, xlim = self.__simulation.t_tracker().init_range(), ylim = self.__simulation.var_trackers()[0].init_range()) # TODO multiple var trackers
        self.ax_var.set_xlabel("Time (seconds)")
        self.ax_var.set_ylabel("Variable")
        self.ax_var.grid()
        self.line_var, = self.ax_var.plot([], [])

        self.__reset()

    # Resets the simulation to its initial conditions
    # Resets all data and labels to default values
    def __reset(self):
        self.__simulation.reset()

        self.line_main.set_data([],[])
        self.time_text_main.set_text('')
        self.energy_text.set_text('')

        self.line_var.set_data([], [])

        # Required for matplotlib to update
        return self.line_main, self.time_text_main, self.energy_text, self.line_var

    def __relim(self, ax_min, ax_max, var_tracker: VariableTracker) -> Tuple[float, float, bool]:
        var_min = var_tracker.min()
        var_max = var_tracker.max()
        var_mean = (var_min + var_max) / 2

        changed = False

        # Check min
        if (var_tracker.min() < ax_min):
            ax_min = var_mean - ((var_mean - var_min) * 2)
            changed = True
        # Check max
        if (var_tracker.max() > ax_max):
            ax_max = var_mean + ((var_max - var_mean) * 2)
            changed = True
        
        return (ax_min, ax_max, changed)

    # Internal function that performs a single animation step
    def __animate(self, i: int, dt: float, draw_dt: float):
        # Simulate next step
        self.__simulation.step_for(dt, draw_dt)

        # Update pendulum position plot
        ((x_0, y_0), (x_1, y_1), (x_2, y_2)) = self.__simulation.pendulum().position_ends()
        x = [x_0, x_1, x_2]
        y = [y_0, y_1, y_2]
        self.line_main.set_data(x, y)

        # Update elapsed time text
        self.time_text_main.set_text('Time = %.1f s' % self.__simulation.elapsed_time())

        # Update energy text
        (potential, kinetic) = self.__simulation.energy()
        total_energy = potential + kinetic
        # self.energy_text.set_text('Potential = %7.3f\nKinetic = %7.3f\nTotal Energy = %7.3f' % (potential, kinetic, total_energy))
        self.energy_text.set_text('Energy = %7.3f' % (total_energy))

        # Update variable tracker plot
        t_tracker = self.__simulation.t_tracker()
        var_tracker = self.__simulation.var_trackers()[0]
        self.line_var.set_data(t_tracker.data, var_tracker.data) # TODO For now we only plot the first variable tracker
        
        # Possibly change axis limits
        (cur_x_min, cur_x_max) = self.ax_var.get_xlim()
        (cur_y_min, cur_y_max) = self.ax_var.get_ylim()
        (x_min, x_max, x_changed) = self.__relim(cur_x_min, cur_x_max, t_tracker)
        (y_min, y_max, y_changed) = self.__relim(cur_y_min, cur_y_max, var_tracker)
        self.ax_var.set_xlim((x_min, x_max))
        self.ax_var.set_ylim((y_min, y_max))
        if (x_changed or y_changed):
            plt.draw()

        # Required for matplotlib to update
        return self.line_main, self.time_text_main, self.energy_text, self.line_var

    # Runs and displays an animation of the pendulum
    #
    #   dt       = time step for the simulation (seconds)
    #   draw_dt  = time between animation frame updates (seconds)
    #   t_final  = time at which the simulation will stop (seconds)
    #
    def run(self, dt: float, draw_dt: float, t_final: float):
        interval = draw_dt * 1000 # interval is in milliseconds
        frames = int(t_final / dt)

        self.ani = animation.FuncAnimation(self.fig, self.__animate, fargs = (dt, draw_dt), frames=frames, interval=interval, blit=True, init_func=self.__reset, repeat=False)

        plt.show()