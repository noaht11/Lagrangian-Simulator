import numpy as np
import sympy as sp
import scipy as sci
from math import sin
from math import cos

sp.init_printing(use_unicode=True)

def dpsidt(t, psi):

    L = 1
    g = 1
    A = 1
    B = 1
    d = 1
    m = 1

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

    A = sp.Matrix([
        [x_theta_1 , y_theta_1 , z_theta_1 , p_theta_1],
        [x_theta_2 , y_theta_2 , z_theta_2 , p_theta_2],
        [x_q       , y_q       , z_q       , p_q      ]
    ])

    (B, _) = A.rref()

    theta_1_dot = B[0,3]
    theta_2_dot = B[1,3]
    q_dot       = B[2,3]

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
    

if __name__ == "__main__":
    psi_init = np.array([0,0,0,0,0,0])
    
    