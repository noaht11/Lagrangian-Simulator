import sympy as sp

from pendulum.core import Coordinates, Physics, MultiPendulum

def n_link_pendulum(n: int, physics: Physics):
    """
    TODO
    """
    t = sp.symbols("t")
    x = sp.Function("x")
    y = sp.Function("y")

    pendulum = None

    thetas = []
    for i in range(n):
        theta = sp.Function("theta_" + str(i + 1)) # 1-index the thetas
        thetas.append(theta)

        if (pendulum is None):
            coordinates = Coordinates(
                x = x,
                y = y,
                theta = theta
            )
            pendulum = MultiPendulum(coordinates, physics)
        else:
            pendulum.attach_pendulum(t, theta, physics)
    
    return (pendulum, t, x, y, thetas)