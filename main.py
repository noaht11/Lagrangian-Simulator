import sys

import sympy as sp
from math import pi

from physics.lagrangian import LagrangianBody, DegreeOfFreedom, Constraint
from physics.pendulum import CompoundPendulumPhysics
from physics.pendulum import SinglePendulumLagrangianPhysics, SinglePendulum, SinglePendulumSolver
from physics.pendulum import n_link_pendulum, MultiPendulum, MultiPendulumSolver
from physics.animation import PhysicsAnimation

sp.init_printing()

def step(text: str):
    print(text, end=""); sys.stdout.flush()

def done():
    print("Done"); sys.stdout.flush()


###################################################################################################################################################################################
# SINGLE PENDULUM
###################################################################################################################################################################################

def single_pendulum_test():
    step("Defining Pendulum...")
    single_pendulum_physics = CompoundPendulumPhysics(
            L = 5,
            m = 3,
            I = 6
        )
        
    t = sp.Symbol("t")
    x = DegreeOfFreedom("x")
    y = DegreeOfFreedom("y")
    theta = DegreeOfFreedom("theta")

    coordinates = SinglePendulumLagrangianPhysics.PendulumCoordinates(x, y, theta)

    pendulum_lagrangian_physics = SinglePendulumLagrangianPhysics(coordinates, single_pendulum_physics, [x, y, theta])
    done()

    step("Constructing Lagrangian Body...")
    pendulum = SinglePendulum(t, pendulum_lagrangian_physics, Constraint(x, sp.S.Zero), Constraint(y, 5.0/20.0*sp.cos(1000*t)))
    done()

    solver = SinglePendulumSolver(pendulum)
    
    step("Generating Simulation...")
    simulation = solver.simulate([pi/4, 0])
    done()

    step("Generating Artist...")
    artist = solver.artist()
    done()

    step("Generating Animation...")
    animation = PhysicsAnimation(simulation, artist)
    done()

    animation.init()
    animation.run(1/50, 1/50, 10000)

###################################################################################################################################################################################
# MULTI PENDULUM
###################################################################################################################################################################################

if __name__ == "__main__":
    print("")

    step("Defining Pendulum...")
    single_pendulum_physics = CompoundPendulumPhysics(
            L = 0.045,
            m = 0.003,
            I = 0
        )
    (pendulum_lagrangian_physics, t, x, y, thetas) = n_link_pendulum(2, single_pendulum_physics)
    done()

    step("Constructing Lagrangian Body...")
    pendulum = MultiPendulum(t, pendulum_lagrangian_physics, Constraint(x, sp.S.Zero), Constraint(y, 0.0025*sp.cos(2*pi*120*t)))
    done()

    solver = MultiPendulumSolver(pendulum)
    
    step("Generating Simulation...")
    simulation = solver.simulate([pi/10, pi/10, 0, 0])
    done()

    step("Generating Artist...")
    artist = solver.artist()
    done()

    step("Generating Animation...")
    animation = PhysicsAnimation(simulation, artist)
    done()

    animation.init()
    animation.run(1/5000, 1/5000, 10000)

    print("")