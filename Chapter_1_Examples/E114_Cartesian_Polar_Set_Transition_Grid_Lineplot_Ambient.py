import numpy as np
import matplotlib.pyplot as plt
from geomotion import manifold as md
from geomotion import utilityfunctions as ut
from geomotion import plottingfunctions as gplt

np.set_printoptions(precision=2)  # Make things print nicely



def polar_to_cartesian(polar_coords):
    cartesian_coords = np.copy(polar_coords)
    cartesian_coords[0] = polar_coords[0] * np.cos(polar_coords[1])
    cartesian_coords[1] = polar_coords[0] * np.sin(polar_coords[1])

    return cartesian_coords


def cartesian_to_polar(cartesian_coords):
    polar_coords = np.copy(cartesian_coords)
    polar_coords[0] = np.sqrt((cartesian_coords[0] * cartesian_coords[0]) + (cartesian_coords[1] * cartesian_coords[1]))
    polar_coords[1] = np.arctan2(cartesian_coords[1], cartesian_coords[0])

    return polar_coords


def ambient_to_cartesian(ambient_coords):
    theta = np.pi/6
    rotmatrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    cartesian_coords = np.squeeze(np.matmul(rotmatrix, ambient_coords[:, None]))
    return cartesian_coords

def cartesian_to_ambient(ambient_coords):
    theta = -np.pi/6
    rotmatrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    cartesian_coords = np.squeeze(np.matmul(rotmatrix, ambient_coords[:, None]))
    return cartesian_coords

transition_table = [[None, cartesian_to_polar, cartesian_to_ambient], [polar_to_cartesian, None, None], [ambient_to_cartesian, None, None]]

R2 = md.Manifold(transition_table, 2)

# Make the working manifold for this problem R2
Q = R2

# Construct the x and y values for a rectangle in the space
q_x = np.concatenate([np.linspace(.5, 2, 20), np.full(20, 2), np.linspace(2, .5, 20), np.full(20, .5)])
q_y = np.concatenate([np.full(20, 0), np.linspace(0, 4, 20), np.full(20, 4), np.linspace(4, 0, 20), ])

# Collect these points into a set
q_numeric = ut.GridArray([q_x, q_y], 1)
q_set_ambient = md.ManifoldElementSet(Q, q_numeric, 2)

q_set_cartesian = q_set_ambient.transition(0)

# Transition the set into polar coordinates
q_set_polar = q_set_cartesian.transition(1)




##############
# Plot the calculated terms
spot_color = gplt.crimson

# Original values
ax_orig = plt.subplot(3, 2, 3)
ax_orig.plot(q_set_cartesian.grid[0], q_set_cartesian.grid[1], color=spot_color)
ax_orig.set_xlim(-.5, 5)
ax_orig.set_ylim(-.5, 5)
ax_orig.set_xticks([0, 1, 2, 3, 4])
ax_orig.set_yticks([0, 1, 2, 3, 4])
ax_orig.set_aspect('equal')
ax_orig.set_axisbelow(True)
ax_orig.grid(True)
ax_orig.axhline(0, color='black', zorder=.75)
ax_orig.axvline(0, color='black', zorder=.75)


# Polar equivalents
ax_polar = plt.subplot(3, 2, 4, projection='polar')
ax_polar.plot(q_set_polar.grid[1], q_set_polar.grid[0], color=spot_color)
ax_polar.set_rlim(0, 6)
ut.convert_polar_xticks_to_radians(ax_polar)

ax_polar = plt.subplot(3, 2, 6)
ax_polar.plot(q_set_polar.grid[1], q_set_polar.grid[0], color=spot_color)
ax_orig.set_xlim(-.5, 5)
ax_orig.set_ylim(-.5, 5)

plt.show()