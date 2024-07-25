from geomotion import rigidbody as rb, kinematicchain as kc, plottingfunctions as gplt
from matplotlib import pyplot as plt
import numpy as np

# Make things print nicely
np.set_printoptions(precision=2)
spot_color = gplt.crimson

G = rb.SE2

# Create a point at the identity
e = G.identity_element()

# Place a chain grounding point at the identity
gp = kc.ground_point(e, .25)

# Create a list of rotational joints and straight links
joints = []
links = []
rot_axis = G.Lie_alg_vector([0, 0, 1])
prismatic_axis = G.Lie_alg_vector([1, 0, 0])
link_transform = G.element([1, 0, 0])

# Start with a simple rotary joint followed by a unit-length link
joints.append(kc.Joint(rot_axis, kc.rotational_joint(.5)))
links.append(kc.Link(link_transform, kc.simple_link(1)))

# Follow with a simple rotary joint followed by a unit-length link
joints.append(kc.Joint(rot_axis, kc.rotational_joint(.5)))
links.append(kc.Link(link_transform, kc.simple_link(1)))

# Then place a prismatic joint and a piston link whose base extension is the midpoint of the link
joints.append(kc.Joint(prismatic_axis, kc.prismatic_joint(.05, .25)))
links.append(kc.Link(link_transform, kc.piston_link(1, .5, .025)))

# Finish with a simple rotary joint followed by a unit-length link
joints.append(kc.Joint(rot_axis, kc.rotational_joint(.5)))
links.append(kc.Link(link_transform, kc.simple_link(1)))

# Form the links, joints, and grounding point into a chain
chain = kc.KinematicChainSequential(links, joints, gp)

# Set the angles in the chain
chain.set_configuration([1, -.5, .25, 1])

# Create a plotting window with equal axes
ax = plt.subplot(1, 1, 1)
ax.set_aspect('equal')

# Draw the chain
chain.draw(ax)

# Put a point at the center of each link
ax.scatter(*chain.link_centers.grid[0:2], color=spot_color)


plt.show()
