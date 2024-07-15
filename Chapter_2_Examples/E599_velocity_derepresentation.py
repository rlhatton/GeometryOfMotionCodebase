from geomotion import representationliegroup as rlgp
from S550_Construct_RxRplus_rep import RxRplus
import numpy as np

np.set_printoptions(precision=2)  # Make things print nicely

def modular_addition_rep(g_value):
    g_radians = g_value[0]
    g_rep = [[np.cos(g_radians), -np.sin(g_radians)],
             [np.sin(g_radians), np.cos(g_radians)]]
    return g_rep


def modular_addition_derep(g_rep):
    g_value = np.arctan2(g_rep[1][0], g_rep[0][0])

    return g_value

print("Modular addition:")

S1plus = rlgp.RepresentationLieGroup(modular_addition_rep, [[1, 0], [0, 1]], modular_addition_derep)

g_circ = S1plus.Lie_alg_vector(2)



print("Representation of g_circ with theta_circ=2 is\n", g_circ.rep, "\n and derepresents back to ", g_circ.value)

g = S1plus.element(np.pi/4)
g_dot = S1plus.vector(g, 2)

print("Representation of g_dot with theta_dot=2 at g=", g, "is\n", g_dot.rep, "\n and derepresents back to ", g_dot.value)


print("\nScale-shift:")


g_circ = RxRplus.Lie_alg_vector([2, -1])



print("Representation of g_circ=[2, -1] is\n", g_circ.rep, "\n and derepresents back to ", g_circ.value)

g = RxRplus.element([.5, 3])
g_dot = RxRplus.vector(g, [1, -0.5])

print("Representation of g_dot=[1, -.5] at g=", g, "is\n", g_dot.rep, "\n and derepresents back to ", g_dot.value)

