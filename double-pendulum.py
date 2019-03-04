import numpy as np
import sympy as sp
from math import sin
from math import cos

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

