from time import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
from scipy.integrate import RK45
from scipy.constants import g
from math import sin
from math import cos
from math import sqrt
from math import pi
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

class ForcedDoublePendulum:

    def __init__(self, L, m, d, force_a, force_b, init_state = [0,0,0,0,0,0], fixed_q = False):
        self.L = L
        self.m = m
        self.d = d
        self.force_a = force_a
        self.force_b = force_b

        self.fixed_q = fixed_q

        self.reset_to(init_state)
    
    def __get_state(self):
        return (self.state[0], self.state[1], self.state[2], self.state[3], self.state[4], self.state[5])

    def __get_state_dot(self):
        return (self.state_dot[0], self.state_dot[1], self.state_dot[2], self.state_dot[3], self.state_dot[4], self.state_dot[5])

    def __neg_pi_to_pi(theta):
        modded = np.mod(theta, 2*pi)
        return modded + (modded > pi) * (-2*pi)

    def __set_state(self, new_state):
        new_state[0] = ForcedDoublePendulum.__neg_pi_to_pi(new_state[0])
        new_state[1] = ForcedDoublePendulum.__neg_pi_to_pi(new_state[1])

        if (self.fixed_q) is True:
            new_state[2] = 0
        
        new_state_dot = self.dstate_dt(state = new_state)
        
        if self.fixed_q is True:
            coeffs = self.__coeff_q(new_state)
            new_state[5] = coeffs[0] * new_state_dot[0] + coeffs[1] * new_state_dot[1]

        self.state = new_state
        self.state_dot = new_state_dot
        
    def reset_default(self):
        self.reset_to(self.init_state)

    def reset_to(self, init_state):        
        self.init_state = init_state
        self.__set_state(init_state)
        self.time_elapsed = 0

        self.solver = RK45(self.dstate_dt, 0, self.init_state, 10000, max_step = (1.0/10))

    def position_ends(self):
        L = self.L
        (theta_1, theta_2, q, p_theta_1, p_theta_2, p_q) = self.__get_state()
        
        x_1 = q + L*sin(theta_1)
        y_1 = 0 + L*cos(theta_1)

        x_2 = q + L*(sin(theta_1) + sin(theta_2))
        y_2 = 0 + L*(cos(theta_1) + cos(theta_2))

        return (q, (x_1, y_1), (x_2, y_2))

    def position_COMs(self):
        L = self.L
        (theta_1, theta_2, q, p_theta_1, p_theta_2, p_q) = self.__get_state()

        x_1 = q + L/2*sin(theta_1)
        y_1 = 0 + L/2*cos(theta_1)

        x_2 = q + L*(sin(theta_1) + 1/2*sin(theta_2))
        y_2 = 0 + L*(cos(theta_1) + 1/2*cos(theta_2))

        return (q, (x_1, y_1), (x_2, y_2))
    
    def energy(self):
        L       = self.L
        m       = self.m
        d       = self.d

        (theta_1, theta_2, q, p_theta_1, p_theta_2, p_q) = self.__get_state()
        (theta_1_dot, theta_2_dot, q_dot, p_theta_1_dot, p_theta_2_dot, p_q_dot) = self.__get_state_dot()

        A       = self.force_a
        B       = self.force_b

        U = 3/2*m*g*L*np.cos(theta_1)   +   1/2*m*g*L*np.cos(theta_2)   +   A/2*m*np.cos(theta_1)   +   B/2*m*np.cos(theta_2)
        T = 1/2*m * (2*q_dot**2   +   q_dot*L*(3*cos(theta_1)*theta_1_dot + cos(theta_2)*theta_2_dot)   +   L**2*(1/4*(5*theta_1_dot**2 + theta_2_dot**2) + theta_1_dot*theta_2_dot*cos(theta_1 - theta_2))   +   d**2*(theta_1_dot**2 + theta_2_dot**2))
        
        return (U, T)

    def __coeff_theta_1(self, state):
        L         = self.L
        m         = self.m
        d         = self.d
        
        (theta_1, theta_2, q, p_theta_1, p_theta_2, p_q) = state

        x_theta_1     = (5*L**2/2 + 2*d**2)
        y_theta_1     = (L**2*cos(theta_1 - theta_2))
        z_theta_1     = (3*L*cos(theta_1))

        coeff_theta_1 = [x_theta_1, y_theta_1, z_theta_1]
        return coeff_theta_1


    def __coeff_theta_2(self, state):
        L         = self.L
        m         = self.m
        d         = self.d
        
        (theta_1, theta_2, q, p_theta_1, p_theta_2, p_q) = state
        
        x_theta_2     = (L**2*cos(theta_1 - theta_2))
        y_theta_2     = (L**2/2 + 2*d**2)
        z_theta_2     = (L*cos(theta_2))

        coeff_theta_2 = [x_theta_2, y_theta_2, z_theta_2]
        return coeff_theta_2

    def __coeff_q(self, state):
        L         = self.L
        m         = self.m
        d         = self.d
        
        (theta_1, theta_2, q, p_theta_1, p_theta_2, p_q) = state
        
        x_q           = (3*L*cos(theta_1))
        y_q           = (L*cos(theta_2))
        z_q           = (4)

        coeff_q       = [x_q, y_q, z_q]
        return coeff_q

    def __coordinates_dot(self, state):
        L         = self.L
        m         = self.m
        d         = self.d
        
        (theta_1, theta_2, q, p_theta_1, p_theta_2, p_q) = state
        
        coeff_theta_1 = self.__coeff_theta_1(state)
        coeff_theta_2 = self.__coeff_theta_2(state)
        coeff_q       = self.__coeff_q(state)

        if self.fixed_q is True:
            coeff_q = [0, 0, 1]

        mat = 1/2*m*np.array([
            coeff_theta_1,
            coeff_theta_2,
            coeff_q
        ])
        p_vec = np.array([p_theta_1, p_theta_2, (p_q if self.fixed_q is not True else 0)])

        mat_inv = np.linalg.inv(mat)
        sol = np.matmul(mat_inv, p_vec)

        return (sol[0], sol[1], sol[2])
        
    def dstate_dt(self, t=0, state=[0,0,0,0,0,0]):
        L         = self.L
        m         = self.m

        (theta_1, theta_2, q, p_theta_1, p_theta_2, p_q) = state

        A         = self.force_a
        B         = self.force_b

        (theta_1_dot, theta_2_dot, q_dot) = self.__coordinates_dot(state)

        p_theta_1_dot = 1/2*m * (-3*q_dot*L*theta_1_dot*sin(theta_1)   -   L**2/4*theta_1_dot*theta_2_dot*sin(theta_1 - theta_2)   +   (3*g*L + A)*sin(theta_1))
        p_theta_2_dot = 1/2*m * (-1*q_dot*L*theta_2_dot*sin(theta_2)   -   L**2/4*theta_1_dot*theta_2_dot*sin(theta_1 - theta_2)   +   (1*g*L + B)*sin(theta_2))
        p_q_dot       = 0

        state_dot = np.array([
            theta_1_dot,
            theta_2_dot,
            q_dot,
            p_theta_1_dot,
            p_theta_2_dot,
            p_q_dot
        ])

        return state_dot

    def step(self, dt):
        #ode_result = odeint(self.dstate_dt, self.state, [0, dt], tfirst = True)[1]
        self.solver.step()
        ode_result = self.solver.y
        self.__set_state(ode_result)
        self.time_elapsed += self.solver.step_size

class PendulumAnimator:
    def __init__(self, pendulum, dt, plot_q = False):
        self.pendulum = pendulum
        self.dt = dt

        self.plot_q = plot_q

    def init(self):
        self.fig = plt.figure(figsize=(8, 8))
        scale_margin_factor_x = 8
        scale_margin_factor_y = 4
        scale_x = (-1 * scale_margin_factor_x * self.pendulum.L, scale_margin_factor_x * self.pendulum.L)
        scale_y = (-1 * scale_margin_factor_y * self.pendulum.L, scale_margin_factor_y * self.pendulum.L)
        self.ax_main = self.fig.add_subplot(211, aspect='equal', autoscale_on=False, xlim=scale_x, ylim=scale_y)
        self.ax_main.grid()

        self.line_main, = self.ax_main.plot([], [], '-', lw=4)
        self.time_text_main = self.ax_main.text(0.02, 0.95, '', transform=self.ax_main.transAxes)
        self.energy_text = self.ax_main.text(0.02, 0.90, '', transform=self.ax_main.transAxes)

        self.ax_q = self.fig.add_subplot(212, autoscale_on=True)
        self.ax_q.set_xlabel('Time (seconds)')
        self.ax_q.set_ylabel('q (metres)')
        self.ax_q.grid()
        
        self.line_q, = self.ax_q.plot([], [])

        self.__reset()

    def __reset(self):
        self.pendulum.reset_default()
        
        if (self.plot_q is True):
            self.t = np.array([0])
            self.theta_1 = np.array([self.pendulum.state[0]])
            self.theta_2 = np.array([self.pendulum.state[1]])
            self.q       = np.array([self.pendulum.state[2]])

        self.line_main.set_data([],[])
        self.time_text_main.set_text('')
        self.energy_text.set_text('')

        self.line_q.set_data([],[])

        # Required for matplotlib to update
        return self.line_main, self.time_text_main, self.energy_text, self.line_q

    def __animate(self, i):
        self.pendulum.step(self.dt)

        if (self.plot_q is True):
            self.t = np.append(self.t, self.t[-1] + self.dt)
            self.q = np.append(self.q, self.pendulum.state[2])
        
        (q, (x_1, y_1), (x_2, y_2)) = self.pendulum.position_ends()
        x = [q, x_1, x_2]
        y = [0, y_1, y_2]
        self.line_main.set_data(x, y)

        self.time_text_main.set_text('time = %.1f s' % self.pendulum.time_elapsed)

        (potential, kinetic) = self.pendulum.energy()
        total_energy = potential + kinetic
        self.energy_text.set_text('potential = %.3f, kinetic = %.3f, total = %.3f' % (potential, kinetic, total_energy))

        if (self.plot_q is True):
            self.line_q.set_data(self.t, self.q)
            self.ax_q.relim()
            self.ax_q.autoscale_view(True,True,True)

        # Required for matplotlib to update
        return self.line_main, self.time_text_main, self.energy_text, self.line_q

    def run(self, frames):
        t0 = time()
        self.__animate(0)
        t1 = time()
        interval = (1000 * self.dt - (t1-t0)) / 20

        self.ani = animation.FuncAnimation(self.fig, self.__animate, frames=frames, interval=interval, blit=True, init_func=self.__reset, repeat=False)

        plt.show()

#######################################################################################################################################################################################

if __name__ == "__main__":
    L = 1 # m
    A = -0 * (3*g*L)
    B = -0 * (1*g*L)
    d = sqrt(1/12)*L
    m = 1 # kg

    dt = 1.0 / 30

    pendulum = ForcedDoublePendulum(L, m, d, A, B, [
        0, # theta_1
        pi, # theta_2
        0, # q
        0, # p_theta_1
        0, # p_theta_2
        0, # p_q
    ], fixed_q = True)

    animator = PendulumAnimator(pendulum, dt, plot_q = False)
    animator.init()
    animator.run(10000)
    