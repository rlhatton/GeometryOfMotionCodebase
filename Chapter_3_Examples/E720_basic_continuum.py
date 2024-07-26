from geomotion import (rigidbody as rb,
                       kinematicchain as kc,
                       continuumbody as cb,
                       plottingfunctions as gplt)
from matplotlib import pyplot as plt
import numpy as np

# Make things print nicely
np.set_printoptions(precision=2)
spot_color = gplt.crimson

G = rb.SE2

# Create a point at the identity
e = G.identity_element()

# Place a chain grounding point at the identity
gp = kc.ground_point(e, .1)

# Make a constant-curvature shape description function
def constant_curvature_continuum(shape_params, s, t):
    kappa = shape_params[0]
    h_circ_a = G.Lie_alg_vector([0, 0, kappa])
    return h_circ_a


# Form the links, joints, and grounding point into a chain
chain = cb.ContinuumBody(constant_curvature_continuum, [0, 1], gp)

# Set the angles in the chain
chain.set_configuration([1])

# Create a plotting window with equal axes
ax = plt.subplot(1, 1, 1)
ax.set_aspect('equal')

# # Draw the chain
chain.draw(ax)



plt.show()
