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
box_width = 3
box_height = 1
box_angle = np.pi/3
box_offset_x = 1
box_offset_y = -.5


edge1_x = np.full(20, box_width/2)
edge2_x = np.linspace(box_width/2, -box_width/2, 20)
edge3_x = np.full(20, -box_width/2)
edge4_x = np.linspace(-box_width/2, box_width/2, 20)
edge_x = np.concatenate([edge1_x, edge2_x, edge3_x, edge4_x]) + 1

edge1_y = np.linspace(-box_height/2, box_height/2, 20)
edge2_y = np.full(20, box_height/2)
edge3_y = np.linspace(box_height/2, -box_height/2, 20)
edge4_y = np.full(20, -box_height/2)
edge_y = np.concatenate([edge1_y, edge2_y, edge3_y, edge4_y]) - 1

edge = np.stack([edge_x, edge_y])

rotmatrix = np.array([[np.cos(box_angle), -np.sin(box_angle)], [np.sin(box_angle), np.cos(box_angle)]])
edge = np.matmul(rotmatrix,edge) + [[box_offset_x], [box_offset_y]]


# Collect these points into a set
q_numeric = ut.GridArray(edge, 1)
q_set_ambient = md.ManifoldElementSet(Q, q_numeric, 2)

q_set_cartesian = q_set_ambient.transition(0)

# Transition the set into polar coordinates
q_set_polar = q_set_cartesian.transition(1)


# For plotting purposes, create a rotated cartesian grid
x = np.linspace(-2, 4, 7)
y = np.linspace(-2, 4, 7)

cart_grid_points = ut.meshgrid_array(x,y)
cart_grid_manifold_elements = md.ManifoldElementSet(R2, cart_grid_points, 0)
cart_grid_rotated_manifold_elements = cart_grid_manifold_elements.transition(2)
cart_grid_rotated_points = cart_grid_rotated_manifold_elements.grid
xg = cart_grid_rotated_points[0]
yg = cart_grid_rotated_points[1]

##############
# Plot the calculated terms
spot_color = gplt.crimson

# ambient values
ax_ambient = plt.subplot(3, 3, 2)
ax_ambient.plot(q_set_ambient.grid[0], q_set_ambient.grid[1], color=spot_color)
ax_ambient.set_xlim(-2, 4)
ax_ambient.set_ylim(-3, 3)
ax_ambient.set_xticks([])  # [-1, 0, 1, 2, 3, 4])
ax_ambient.set_yticks([])  # [-1, 0, 1, 2, 3, 4])
ax_ambient.set_aspect('equal')
ax_ambient.set_axisbelow(True)
# ax_ambient.grid(True)
# ax_ambient.axhline(0, color='black', zorder=.75)
# ax_ambient.axvline(0, color='black', zorder=.75)


# Cartesian grid on ambient space
ax_cart = plt.subplot(3, 3, 4)
ax_cart.plot(q_set_ambient.grid[0], q_set_ambient.grid[1], color=spot_color)
ax_cart.pcolormesh(xg, yg, np.zeros([xg.shape[0]-1, xg.shape[1]-1]), edgecolor='grey', facecolor='none', linewidth=0.25)
ax_cart.plot(xg[2], yg[2], color='black')
ax_cart.plot(xg.T[2], yg.T[2], color='black')
ax_cart.set_xlim(-2, 4)
ax_cart.set_ylim(-3, 3)
ax_cart.set_xticks([])  # [-1, 0, 1, 2, 3, 4])
ax_cart.set_yticks([])  # [-1, 0, 1, 2, 3, 4])
ax_cart.set_aspect('equal')
ax_cart.set_axisbelow(True)
# ax_cart.grid(True)
# ax_cart.axhline(0, color='black', zorder=.75)
# ax_cart.axvline(0, color='black', zorder=.75)


# Cartesian values
ax_cart_chart = plt.subplot(3, 3, 7)
ax_cart_chart.plot(q_set_cartesian.grid[0], q_set_cartesian.grid[1], color=spot_color)
ax_cart_chart.set_xlim(-2, 2)
ax_cart_chart.set_ylim(-2, 2)
ax_cart_chart.set_xticks([-1, 0, 1, 2, 3, 4])
ax_cart_chart.set_yticks([-1, 0, 1, 2, 3, 4])
ax_cart_chart.set_aspect('equal')
ax_cart_chart.set_axisbelow(True)
ax_cart_chart.grid(True)
ax_cart_chart.axhline(0, color='black', zorder=.75)
ax_cart_chart.axvline(0, color='black', zorder=.75)


# Polar equivalents
ax_polar = plt.subplot(3, 3, 6, projection='polar')
ax_polar.plot(q_set_polar.grid[1], q_set_polar.grid[0], color=spot_color)
ax_polar.set_rlim(0, 6)
ut.convert_polar_xticks_to_radians(ax_polar)

ax_polar_chart = plt.subplot(3, 3, 9)
ax_polar_chart.plot(q_set_polar.grid[1], q_set_polar.grid[0], color=spot_color)
# ax_polar_chart.set_xlim(-.5, 5)
# ax_polar_chart.set_ylim(-.5, 5)

plt.show()