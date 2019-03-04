import numpy as np
from sympy import Matrix
from sympy import N
from scipy.integrate import solve_ivp
from math import sin
from math import cos
from math import sqrt
from math import pi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

L = 0.15 # m
g = 9.81 # m/s^2
A = -10 * (3*g*L)
B = -10 * (1*g*L)
d = sqrt(1/12)*L
m = 0.25 # kg

def dpsidt(t, psi):
    theta_1   = psi[0]
    theta_2   = psi[1]
    q         = psi[2]
    p_theta_1 = psi[3]
    p_theta_2 = psi[4]
    p_q       = psi[5]

    x_theta_1 = 1/2*m * (5*L**2/2 + 2*d**2)
    y_theta_1 = 1/2*m * (L**2*cos(theta_1 - theta_2))
    z_theta_1 = 1/2*m * (3*L*cos(theta_1))

    x_theta_2 = 1/2*m * (L**2*cos(theta_1 - theta_2))
    y_theta_2 = 1/2*m * (L**2/2 + 2*d**2)
    z_theta_2 = 1/2*m * (L*cos(theta_2))

    x_q       = 1/2*m * (3*L*cos(theta_1))
    y_q       = 1/2*m * (L*cos(theta_2))
    z_q       = 1/2*m * (4)

    mat = Matrix([
        [x_theta_1 , y_theta_1 , z_theta_1 , p_theta_1],
        [x_theta_2 , y_theta_2 , z_theta_2 , p_theta_2],
        [x_q       , y_q       , z_q       , p_q      ]
    ])

    (sol, _) = mat.rref()

    theta_1_dot = float(sol[0,3])
    theta_2_dot = float(sol[1,3])
    q_dot       = float(sol[2,3])

    p_theta_1_dot = 1/2*m * (-3*q_dot*L*theta_1_dot*sin(theta_1)   -   L**2/4*theta_1_dot*theta_2_dot*sin(theta_1 - theta_2)   +   (3*g*L + A)*sin(theta_1))
    p_theta_2_dot = 1/2*m * (-1*q_dot*L*theta_2_dot*sin(theta_2)   -   L**2/4*theta_1_dot*theta_2_dot*sin(theta_1 - theta_2)   +   (1*g*L + B)*sin(theta_2))
    p_q_dot       = 0

    psi_dot = np.array([
        theta_1_dot,
        theta_2_dot,
        q_dot,
        p_theta_1_dot,
        p_theta_2_dot,
        p_q_dot
    ])

    return psi_dot

def potential_plot():
    num_points = 100
    theta_1 = np.tile(np.linspace(-pi, pi, num_points), (num_points, 1))
    theta_2 = np.transpose(theta_1)

    U = 3/2*m*g*L*np.cos(theta_1)   +   1/2*m*g*L*np.cos(theta_2)   +   A/2*m*np.cos(theta_1)   +   B/2*m*np.cos(theta_2)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(theta_1, theta_2, U)

    plt.show()

def solve(psi_init, t_domain):
    result = solve_ivp(dpsidt, t_domain, psi_init)

    num_points = result.y.shape[1]
    
    print("\n\nNumber of data points: %d\n\n" % num_points)

    theta_1 = np.mod(result.y[0, :], 2*pi)
    theta_2 = np.mod(result.y[1, :], 2*pi)
    q       = result.y[2, :]

    x_1 = q + L/2*np.sin(theta_1)
    y_1 = 0 + L/2*np.cos(theta_1)

    fig_1 = plt.figure()
    ax_1 = fig_1.add_subplot(111, projection='3d')
    ax_1.plot(x_1, y_1, result.t)

    x_2 = q + L*(np.sin(theta_1) + 1/2*np.sin(theta_2))
    y_2 = 0 + L*(np.cos(theta_1) + 1/2*np.cos(theta_2))
    
    # fig_2 = plt.figure()
    # ax_2 = fig_2.add_subplot(111, projection='3d')
    # ax_2.plot(x_2, y_2, result.t)

    fig_theta_1 = plt.figure()
    ax_theta_1 = fig_theta_1.add_subplot(111)
    ax_theta_1.plot(result.t, np.transpose(theta_1))
    ax_theta_1.set_title("Theta 1")

    fig_theta_2 = plt.figure()
    ax_theta_2 = fig_theta_2.add_subplot(111)
    ax_theta_2.plot(result.t, np.transpose(theta_2))
    ax_theta_2.set_title("Theta 2")

    fig_q = plt.figure()
    ax_q = fig_q.add_subplot(111)
    ax_q.plot(result.t, np.transpose(q))
    ax_q.set_title("q")

    plt.show()

if __name__ == "__main__":
    psi_init = np.array([pi/20,0,0,0,0,0])
    t_domain = [0, 2]

    solve(psi_init, t_domain)