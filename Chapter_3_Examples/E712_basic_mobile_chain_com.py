from geomotion import rigidbody as rb, kinematicchain as kc, plottingfunctions as gplt
from matplotlib import pyplot as plt
import numpy as np

# Make things print nicely
np.set_printoptions(precision=2)
spot_color = gplt.crimson

G = rb.SE2

# Create a list of two rotational joints and two straight links
joints = []
links = []
rot_axis = G.Lie_alg_vector([0, 0, 1])
link_transform = G.element([1, 0, 0])
for j in range(2):
    joints.append(kc.Joint(rot_axis, kc.rotational_joint(.5)))
for l in range(3):
    links.append(kc.Link(link_transform, kc.simple_link(1)))

# Form the links, joints, and grounding point into a mobile chain
chain = kc.KinematicChainMobileSequential(links, joints, 'com')


# Create a plotting window with equal axes
ax = plt.subplot(1, 1, 1)
ax.set_aspect('equal')

for x in [-5, 0, 5]:
    for y in [-5, 0, 5]:
        g = G.element([x, y, 0])

        # Set the angles in the chain
        chain.set_configuration(g, [x/5, y/5])


        # Draw the chain
        chain.draw(ax)




plt.show()
