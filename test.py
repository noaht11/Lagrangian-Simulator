from double_pendulum import ForcedDoublePendulum

from math import cos, sin, sqrt, pi

from scipy.constants import g

def calc_theta_1_dot(state, m, L):
    (theta_1, theta_2, q, p_theta_1, p_theta_2, p_q) = state

    return 6 / (m*L**2) * (2*p_theta_1 - 3*cos(theta_1 - theta_2)*p_theta_2) / (16 - 9*(cos(theta_1 - theta_2))**2)

def calc_theta_2_dot(state, m, L):
    (theta_1, theta_2, q, p_theta_1, p_theta_2, p_q) = state

    return 6 / (m*L**2) * (8*p_theta_2 - 3*cos(theta_1 - theta_2)*p_theta_1) / (16 - 9*(cos(theta_1 - theta_2))**2)

def calc_p_theta_1_dot(state, m, L):
    (theta_1, theta_2, q, p_theta_1, p_theta_2, p_q) = state
    theta_1_dot = calc_theta_1_dot(state, m, L)
    theta_2_dot = calc_theta_2_dot(state, m, L)

    return -1/2*m*L**2 * (theta_1_dot*theta_2_dot*sin(theta_1 - theta_2) + 3*g/L*sin(theta_1))

def calc_p_theta_2_dot(state, m, L):
    (theta_1, theta_2, q, p_theta_1, p_theta_2, p_q) = state
    theta_1_dot = calc_theta_1_dot(state, m, L)
    theta_2_dot = calc_theta_2_dot(state, m, L)

    return -1/2*m*L**2 * (-theta_1_dot*theta_2_dot*sin(theta_1 - theta_2) + g/L*sin(theta_2))

L = 1
m = 1
d = sqrt(1/12)*L

# init_state = [
#     pi/4, # theta_1
#     pi/4, # theta_2
#     0, # q
#     0, # p_theta_1
#     0, # p_theta_2
#     0, # p_q
# ]

init_state_wiki = [
    -2.22943232, # theta_1
    -0.8263203, # theta_2
    0, # q
    -5.90443021, # p_theta_1
    -2.65990268, # p_theta_2
    2.59887516, # p_q
]
init_state = [
    pi - -2.22943232, # theta_1
    pi - -0.8263203, # theta_2
    0, # q
    -5.90443021, # p_theta_1
    -2.65990268, # p_theta_2
    2.59887516, # p_q
]

pendulum = ForcedDoublePendulum(L, m, d, 0, 0, init_state, fixed_q = True)

theta_1_dot = calc_theta_1_dot(init_state_wiki, m, L)
theta_2_dot = calc_theta_2_dot(init_state_wiki, m, L)
p_theta_1_dot = calc_p_theta_1_dot(init_state_wiki, m, L)
p_theta_2_dot = calc_p_theta_2_dot(init_state_wiki, m, L)

print("\n\n")

print(theta_1_dot)
print(theta_2_dot)
print(p_theta_1_dot)
print(p_theta_2_dot)

print("\n\n")

print(pendulum.dstate_dt(0, init_state))

print("\n\n")
