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


# Define a phase-amplitude to joint-angle reparameterization function
def phase_amp_reparam(phase_amp):
    A = phase_amp[0]
    phi = phase_amp[1]
    joint_angles = [A * np.cos(phi), A * np.sin(phi)]
    return joint_angles


# Form the links, joints, and grounding point into a mobile chain
chain = kc.KinematicChainMobileSequential(links, joints, 'midpoint', 1, phase_amp_reparam)

# Create a plotting window with equal axes
ax = plt.subplot(1, 1, 1)
ax.set_aspect('equal')

for phi in [np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]:
    for A in [0, np.sqrt(2), 2* np.sqrt(2)]:
        g = G.element([phi*2, A, 0])

        # Set the angles in the chain
        chain.set_configuration(g, [A, phi])

        # Draw the chain
        chain.draw(ax, False)

plt.show()
