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
for j in range(3):
    joints.append(kc.Joint(rot_axis, kc.rotational_joint(.5)))
for l in range(4):
    links.append(kc.Link(link_transform, kc.simple_link(1)))


# Define an even-odd reparameterization function
def four_link_reparam(two_modes):
    modal_matrix = np.array([[1, -1], [1, 0], [1, 1]])

    joint_angles = np.matmul(modal_matrix, two_modes)
    return joint_angles


# Form the links, joints, and grounding point into a mobile chain
chain = kc.KinematicChainMobileSequential(links, joints, 'midpoint', 1, four_link_reparam)

# Create a plotting window with equal axes
ax = plt.subplot(1, 1, 1)
ax.set_aspect('equal')

for x in [-1, 0, 1]:
    for y in [-1, 0, 1]:
        g = G.element([x*5, y*5, 0])

        # Set the angles in the chain
        chain.set_configuration(g, [x, y])

        # Draw the chain
        chain.draw(ax, False)

plt.show()
