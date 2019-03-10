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

# Represents a double pendulum with an artificial potential applied and the pivot point free to move along the horizontal axis
class ForcedDoublePendulum:

    # L             =   length of each arm
    # m             =   mass of each arm
    # d             =   moment of inertia factor such that I = 1/2 * m * d^2
    # A             =   forcing factor for theta_1: a potential of 1/2*m * (-A*3*g*L) * cos(theta_1) will be applied
    #                     a value of 1 will exactly cancel the gravitation potential for the first arm
    #                     a value of 0 will leave the first arm unforced
    # B             =   forcing factor for theta_2: a potential of 1/2*m * (-B*1*g*L) * cos(theta_2) will be applied
    #                     a value of 1 will exactly cancel the gravitation potential for the second arm
    #                     a value of 0 will leave the second arm unforced
    # init_state    =   initial state of the pendulum as an array of six values:
    #                     [
    #                       theta_1,
    #                       theta_2,
    #                       q,
    #                       p_theta_1,
    #                       p_theta_2,
    #                       p_q,
    #                     ]
    # solver        =   Method for solving the ODEs, must be one of:
    #                       - "ODEINT"
    #                       - "RK45"
    # rk45_max_dt   =   Only required if using RK45 solver - maximum time step to use - dt passed to step() will be ignored
    # rk45_t_bound  =   Only required if using RK45 solver - time at which the solver will stop (also determines direction of integration)
    #
    def __init__(self, L, m, d, A, B, init_state = [0,0,0,0,0,0], solver = "ODEINT", rk45_max_dt = 1, rk45_t_bound = 0, fixed_q = False):
        self.L = L
        self.m = m
        self.d = d
        self.force_a = -A * 3*g*L
        self.force_b = -B * 1*g*L

        self.solver = "ODEINT"

        self.rk45_max_dt = rk45_max_dt
        self.rk45_t_bound = rk45_t_bound

        self.fixed_q = fixed_q

        self.reset_to(init_state)
    
    # Returns the elements of the current state extracted into a tuple of 6 elements (in the same order as the array)
    def get_state(self):
        return (self.__state[0], self.__state[1], self.__state[2], self.__state[3], self.__state[4], self.__state[5])

    def theta_1(self)   : return self.__state[0]
    def theta_2(self)   : return self.__state[1]
    def q(self)         : return self.__state[2]
    def p_theta_1(self) : return self.__state[3]
    def p_theta_2(self) : return self.__state[4]
    def p_q(self)       : return self.__state[5]

    # Returns the elements of the time derivative of the current state extracted into a tuple of 6 elements (in the same order as the array)
    def get_state_dot(self):
        return (self.__state_dot[0], self.__state_dot[1], self.__state_dot[2], self.__state_dot[3], self.__state_dot[4], self.__state_dot[5])

    # Converts an angle in radians to the equivalent angle in radians constrained between -pi and pi, where 0 remains at angle 0
    def __neg_pi_to_pi(theta):
        modded = np.mod(theta, 2*pi)
        return modded + (modded > pi) * (-2*pi)

    # Update the state of the pendulum
    # The self.__state variable should never be directly modified since self.__state_dot must also be updated when self.__state is
    # Instead always use this method which automatically calculates the time derivative of the state to update the self.__state_dot as well
    def __set_state(self, new_state):
        new_state[0] = ForcedDoublePendulum.__neg_pi_to_pi(new_state[0])
        new_state[1] = ForcedDoublePendulum.__neg_pi_to_pi(new_state[1])

        if (self.fixed_q) is True:
            new_state[2] = 0
        
        new_state_dot = self.dstate_dt(state = new_state)
        
        if self.fixed_q is True:
            coeffs = self.__coeff_q(new_state)
            new_state[5] = coeffs[0] * new_state_dot[0] + coeffs[1] * new_state_dot[1]

        self.__state = new_state
        self.__state_dot = new_state_dot
        
    # Resets the state to the initial state that was most recently passed to either the constructor or reset_to(init_state)
    def reset_default(self):
        self.reset_to(self.init_state)

    # Resets the state to the provided init_state and saves it for future calls to reset_default()
    # Also resets the elapsed time to 0
    def reset_to(self, init_state):        
        self.init_state = init_state
        self.__set_state(init_state)
        self.time_elapsed = 0

        self.solver_rk45 = RK45(self.dstate_dt, 0, self.init_state, self.rk45_t_bound, max_step = self.rk45_max_dt)

    # Calculates and returns the positions of the endpoints of the arms of the pendulum
    #
    # The return value is a tuple of the following form:
    #   (
    #       (pivot_x, pivot_y),
    #       (joint_x, joint_y),
    #       (end_x  , end_y  )
    #   )
    def position_ends(self):
        L = self.L
        (theta_1, theta_2, q, p_theta_1, p_theta_2, p_q) = self.get_state()
        
        x_1 = q + L*sin(theta_1)
        y_1 = 0 + L*cos(theta_1)

        x_2 = q + L*(sin(theta_1) + sin(theta_2))
        y_2 = 0 + L*(cos(theta_1) + cos(theta_2))

        return ((q, 0), (x_1, y_1), (x_2, y_2))

    # Calculates and returns the positions of the pivot point and COM (centre of mass) of each arm of the pendulum
    #
    # The return value is a tuple of the following form:
    #   (
    #       (pivot_x, pivot_y),
    #       (COM_1_x, COM_1_y),
    #       (COM_2_x, COM_2_y)
    #   )
    def position_COMs(self):
        L = self.L
        (theta_1, theta_2, q, p_theta_1, p_theta_2, p_q) = self.get_state()

        x_1 = q + L/2*sin(theta_1)
        y_1 = 0 + L/2*cos(theta_1)

        x_2 = q + L*(sin(theta_1) + 1/2*sin(theta_2))
        y_2 = 0 + L*(cos(theta_1) + 1/2*cos(theta_2))

        return ((q, 0), (x_1, y_1), (x_2, y_2))
    
    # Calculates the potential and kinetic energy of the current state of the pendulum
    #
    # Return value is a tuple of the form:
    #   (potential_energy, kinetic_energy)
    def energy(self):
        L       = self.L
        m       = self.m
        d       = self.d

        (theta_1, theta_2, q, p_theta_1, p_theta_2, p_q) = self.get_state()
        (theta_1_dot, theta_2_dot, q_dot, p_theta_1_dot, p_theta_2_dot, p_q_dot) = self.get_state_dot()

        A       = self.force_a
        B       = self.force_b

        U = 3/2*m*g*L*cos(theta_1)   +   1/2*m*g*L*cos(theta_2)   +   A/2*m*cos(theta_1)   +   B/2*m*cos(theta_2)
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
        
        # print("\n\ntheta1 = %.5f\ntheta2 = %.5f\n" % (theta_1, theta_2))

        A         = self.force_a
        B         = self.force_b

        (theta_1_dot, theta_2_dot, q_dot) = self.__coordinates_dot(state)

        p_theta_1_dot = 1/2*m * (-3*q_dot*L*theta_1_dot*sin(theta_1)   -   L**2*theta_1_dot*theta_2_dot*sin(theta_1 - theta_2)   +   (3*g*L + A)*sin(theta_1))
        p_theta_2_dot = 1/2*m * (-1*q_dot*L*theta_2_dot*sin(theta_2)   +   L**2*theta_1_dot*theta_2_dot*sin(theta_1 - theta_2)   +   (1*g*L + B)*sin(theta_2))
        p_q_dot       = 0

        # print("  theta1_dot = %.6f" % p_theta_1_dot)
        # print("  theta2_dot = %.6f" % p_theta_2_dot)
        # print("  p_theta1_dot = %.6f" % p_theta_1_dot)
        # print("  p_theta2_dot = %.6f" % p_theta_2_dot)

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
        ode_result = odeint(self.dstate_dt, self.__state, [0, dt], tfirst = True)[1]
        print(ode_result)
        self.time_elapsed += dt
        # self.solver.step()
        # ode_result = self.solver.y
        # self.time_elapsed += self.solver.step_size
        self.__set_state(ode_result)

class PendulumAnimator:
    def __init__(self, pendulum, dt, plot_q = False):
        self.pendulum = pendulum
        self.dt = dt

        self.plot_q = plot_q

    def init(self):
        self.fig = plt.figure(figsize=(8, 8))
        scale_margin_factor_x = 6
        scale_margin_factor_y = 3
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

            self.theta_1 = np.array(self.pendulum.theta_1())
            self.theta_2 = np.array(self.pendulum.theta_2())
            self.q       = np.array(self.pendulum.q())

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
            self.q = np.append(self.q, self.pendulum.q())
        
        ((x_0, y_0), (x_1, y_1), (x_2, y_2)) = self.pendulum.position_ends()
        x = [x_0, x_1, x_2]
        y = [y_0, y_1, y_2]
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
        interval = (1000 * self.dt - (t1-t0)) / 1

        self.ani = animation.FuncAnimation(self.fig, self.__animate, frames=frames, interval=interval, blit=True, init_func=self.__reset, repeat=False)

        plt.show()

#######################################################################################################################################################################################

if __name__ == "__main__":
    L = 1 # m
    A = 0
    B = 0
    d = sqrt(1/12)*L
    m = 2 # kg

    dt = 1.0 / 50

    pendulum = ForcedDoublePendulum(L, m, d, A, B, [
        pi/2, # theta_1
        pi/2, # theta_2
        0, # q
        0, # p_theta_1
        0, # p_theta_2
        0, # p_q
    ], fixed_q = False)

    animator = PendulumAnimator(pendulum, dt, plot_q = False)
    animator.init()
    animator.run(10)
    