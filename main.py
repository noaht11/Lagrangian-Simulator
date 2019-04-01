import sys

import sympy as sp
from math import pi

from physics.lagrangian import LagrangianBody, DegreeOfFreedom, Constraint
from physics.pendulum import SinglePendulum, SinglePendulumLagrangianPhysics, SinglePendulumArtist, SinglePendulumSolver, CompoundPendulumPhysics
from physics.animation import PhysicsAnimation

sp.init_printing()

def step(text: str):
    print(text, end=""); sys.stdout.flush()

def done():
    print("Done"); sys.stdout.flush()

if __name__ == "__main__":
    print("")

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

    pendulum_lagrangian_physics = SinglePendulumLagrangianPhysics(coordinates, single_pendulum_physics)
    # (pendulum, t, x, y, thetas) = n_link_pendulum(2, pendulum_physics)
    done()

    step("Constructing Lagrangian Body...")
    pendulum = SinglePendulum(t, pendulum_lagrangian_physics, Constraint(x, sp.S.Zero), Constraint(y, sp.S.Zero))
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

    # step("Calculating Lagrangian...")
    # L = body.lagrangian()
    # done()

    # step("Generating Symbolic ODE Equations...")
    # odeExpressions = L.solve()
    # done()

    # step("Converting to Numerically Solvable Equations...")
    # solver = LagrangianNumericalSolver.from_ode_expr(odeExpressions)
    # done()

    # state = [
    #     0,
    #     pi/2,
    #     pi/4,
    #     0,
    #     2,
    #     3
    # ]

    # print(solver.state_to_y(0, state))

    print("")