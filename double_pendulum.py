from time import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
from scipy.constants import g
from math import sin
from math import cos
from math import sqrt
from math import pi
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

class ForcedDoublePendulum:

    def __init__(self, L, m, d, force_a, force_b, init_state = [0,0,0,0,0,0]):
        self.L = L
        self.m = m
        self.d = d
        self.force_a = force_a
        self.force_b = force_b

        self.reset(init_state)
    
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
        self.state = new_state
        self.state_dot = self.dstate_dt(self.state)

    def reset(self, init_state):
        self.__set_state(init_state)
        self.time_elapsed = 0

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

    def __coordinates_dot(self, state):
        L         = self.L
        m         = self.m
        d         = self.d
        
        (theta_1, theta_2, q, p_theta_1, p_theta_2, p_q) = state
        
        x_theta_1 = (5*L**2/2 + 2*d**2)
        y_theta_1 = (L**2*cos(theta_1 - theta_2))
        z_theta_1 = (3*L*cos(theta_1))

        x_theta_2 = (L**2*cos(theta_1 - theta_2))
        y_theta_2 = (L**2/2 + 2*d**2)
        z_theta_2 = (L*cos(theta_2))

        x_q       = (3*L*cos(theta_1))
        y_q       = (L*cos(theta_2))
        z_q       = (4)

        mat = 1/2*m*np.array([
            [x_theta_1 , y_theta_1 , z_theta_1],
            [x_theta_2 , y_theta_2 , z_theta_2],
            [x_q       , y_q       , z_q      ]
        ])
        p_vec = np.array([p_theta_1, p_theta_2, p_q])

        mat_inv = np.linalg.inv(mat)
        sol = np.matmul(mat_inv, p_vec)

        return (sol[0], sol[1], sol[2])
        
    def dstate_dt(self, state, t=0):
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
        ode_result = odeint(self.dstate_dt, self.state, [0, dt])[1]
        self.__set_state(ode_result)
        self.time_elapsed += dt

class PendulumAnimator:
    def __init__(self, pendulum, dt):
        self.pendulum = pendulum
        self.dt = dt

    def init(self):
        self.fig = plt.figure()
        scale_margin_factor = 3
        scale = (-1 * scale_margin_factor * self.pendulum.L, scale_margin_factor * self.pendulum.L)
        self.ax = self.fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=scale, ylim=scale)
        self.ax.grid()

        self.line, = self.ax.plot([],[], 'o-', lw=2)
        self.time_text = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes)
        self.energy_text = self.ax.text(0.02, 0.90, '', transform=self.ax.transAxes)

        self.__reset()

    def __reset(self):
        self.line.set_data([],[])
        self.time_text.set_text('')
        self.energy_text.set_text('')

        # Required for matplotlib to update
        return self.line, self.time_text, self.energy_text

    def __animate(self, i):
        self.pendulum.step(self.dt)
        
        (q, (x_1, y_1), (x_2, y_2)) = self.pendulum.position_ends()
        x = [q, x_1, x_2]
        y = [0, y_1, y_2]
        self.line.set_data(x, y)

        self.time_text.set_text('time = %.1f s' % self.pendulum.time_elapsed)

        (potential, kinetic) = self.pendulum.energy()
        total_energy = potential + kinetic
        self.energy_text.set_text('potential = %.3f, kinetic = %.3f, total = %.3f' % (potential, kinetic, total_energy))

        # Required for matplotlib to update
        return self.line, self.time_text, self.energy_text

    def run(self, frames):
        t0 = time()
        self.__animate(0)
        t1 = time()
        interval = 1000 * self.dt - (t1-t0)

        self.ani = animation.FuncAnimation(self.fig, self.__animate, frames=frames, interval=interval, blit=True, init_func=self.__reset)

        plt.show()

if __name__ == "__main__":
    L = 1 # m
    A = -10 * (3*g*L)
    B = -10 * (1*g*L)
    d = sqrt(1/12)*L
    m = 1 # kg

    dt = 1.0 / 50

    pendulum = ForcedDoublePendulum(L, m, d, A, B, [
        pi/10, # theta_1
        0, # theta_2
        0, # q
        0, # p_theta_1
        0, # p_theta_2
        0, # p_q
    ])

    animator = PendulumAnimator(pendulum, dt)
    animator.init()
    animator.run(300)