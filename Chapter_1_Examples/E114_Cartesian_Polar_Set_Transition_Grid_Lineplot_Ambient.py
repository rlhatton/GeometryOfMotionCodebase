import numpy as np
import matplotlib.pyplot as plt
from geomotion import manifold as md
from geomotion import utilityfunctions as ut
from geomotion import plottingfunctions as gplt
from S100_Construct_R2 import R2


##########
# Set up the transition map for R2 with the standard polar and Cartesian charts


# Define functions for embedding the Cartesian chart in an R2 ambient space, whose coordinates
# for computation are rotated relative to the Cartesian chart
def ambient_to_cartesian(ambient_coords):
    ambient_coords = np.array(ambient_coords)
    theta_amb = np.pi / 3
    rotmatrix_amb = np.array([[np.cos(theta_amb), -np.sin(theta_amb)], [np.sin(theta_amb), np.cos(theta_amb)]])
    cartesian_coords = np.squeeze(np.matmul(rotmatrix_amb, ambient_coords[:, None]))
    return cartesian_coords


def cartesian_to_ambient(cartesian_coords):
    cartesian_coords = np.array(cartesian_coords)
    theta_cart = -np.pi / 3
    rotmatrix_cart = np.array([[np.cos(theta_cart), -np.sin(theta_cart)], [np.sin(theta_cart), np.cos(theta_cart)]])
    ambient_coords = np.squeeze(np.matmul(rotmatrix_cart, cartesian_coords[:, None]))
    return ambient_coords


# Make the working manifold for this problem R2
Q = R2

# Also make the ambient space a (second copy of) R2
Q_amb = R2

# Generate parameterization and embedding mappings between the ambient space
# and the working manifold
param_map = md.ManifoldMap(Q_amb, Q, ambient_to_cartesian, 0, 0)
embed_map = md.ManifoldMap(Q, Q_amb, cartesian_to_ambient, 0, 0)

#####################
# Construct the points describing a rotated rectangle in the ambient space

# Specify the rectangle's geometry in the ambient-space chart
box_width = 1
box_height = 3
box_angle = -np.pi / 3
box_offset_x = 2
box_offset_y = .5

# Build the edges of the box counterclockwise from the lower-right corner
edge1_x = np.full(20, box_width / 2)
edge2_x = np.linspace(box_width / 2, -box_width / 2, 20)
edge3_x = np.full(20, -box_width / 2)
edge4_x = np.linspace(-box_width / 2, box_width / 2, 20)
edge_x = np.concatenate([edge1_x, edge2_x, edge3_x, edge4_x]) + 1

edge1_y = np.linspace(-box_height / 2, box_height / 2, 20)
edge2_y = np.full(20, box_height / 2)
edge3_y = np.linspace(box_height / 2, -box_height / 2, 20)
edge4_y = np.full(20, -box_height / 2)
edge_y = np.concatenate([edge1_y, edge2_y, edge3_y, edge4_y]) - 1

# Combine the x and y edges into a single 2xN array
edge = np.stack([edge_x, edge_y])

# Construct a rotation matrix corresponding to the box angle
rotmatrix = np.array([[np.cos(box_angle), -np.sin(box_angle)], [np.sin(box_angle), np.cos(box_angle)]])

# Rotate the box and apply the offset
edge = np.matmul(rotmatrix, edge) + [[box_offset_x], [box_offset_y]]

########
# Collect these points into a ManifoldElementSet

# This specifies that the first dimension is the coordinate values, and the remaining
# dimensions are for different points
q_numeric = ut.GridArray(edge, 1)

#  Turn the data into ManifoldElements as specified in the ambient space
q_set_ambient = md.ManifoldElementSet(Q, q_numeric, 0)

# Convert the rectangle into a set of points specified in the Cartesian chart
q_set_cartesian = param_map(q_set_ambient)

# Convert the rectangle into a set of points specified in the polar chart
q_set_polar = q_set_cartesian.transition(1)

############
# For plotting purposes, create a rotated cartesian grid and a polar grid

# Generate a loosely spaced set of points on two axes
x = np.linspace(-2, 4, 7)
y = np.linspace(-2, 4, 7)

# Build a meshgrid from x and y, using the meshgrid_array function, which generates a GridArray with the
# n_outer value generated automatically
cart_grid = ut.meshgrid_array(x, y)

# Turn the meshgrid into a ManifoldElementSet, specified in the Cartesian chart
ambient_cartesian_points = md.ManifoldElementSet(Q_amb, cart_grid, 0)

# Turn the meshgrid into a ManifoldElementSet, specified in the Cartesian chart on the manifold
manifold_cartesian_points = md.ManifoldElementSet(Q, cart_grid, 0)

# Convert (i.e. rotate) the grid into the ambient-space coordinates
manifold_cartesian_points_amb = embed_map(manifold_cartesian_points)

##
# Generate a loosely-spaced set of points for polar axes
r = np.linspace(.25, 4, 7)
theta = np.linspace(-3, 3, 21)

# Build a meshgrid from r and theta, using the meshgrid_array function, which generates a GridArray with the
# n_outer value generated automatically
polar_grid = ut.meshgrid_array(r, theta)

# Turn the meshgrid into a ManifoldElementSet, specified in the polar chart
polar_grid_points = md.ManifoldElementSet(R2, polar_grid, 1)

# Get the ambient locations of the polar grid points
polar_grid_points_amb = embed_map(polar_grid_points)
##############
# Plot the calculated terms

# Use my red color for the plots
spot_color = gplt.crimson

###
# Plot the rectangle in the ambient space, with no coordinate grid (because the ambient space in principle is
# coordinate-free, and we've only supplied it with coordinates so we can actually tell the computer to plot things
ax_ambient = plt.subplot(3, 4, 2)
ax_ambient.plot(*q_set_ambient.grid, color=spot_color)
ax_ambient.set_xlim(-2, 4)
ax_ambient.set_ylim(-3, 3)
ax_ambient.set_xticks([])
ax_ambient.set_yticks([])
ax_ambient.set_aspect('equal')
ax_ambient.set_axisbelow(True)

# Plot the Cartesian grid and the rectangle as they appear in ambient space
ax_cart = plt.subplot(3, 4, 5)
ax_cart.plot(*q_set_ambient.grid, color=spot_color)
ax_cart.pcolormesh(*manifold_cartesian_points_amb.grid,
                   np.zeros([manifold_cartesian_points_amb.shape[0] - 1, manifold_cartesian_points_amb.shape[1] - 1]),
                   edgecolor='grey',
                   facecolor='none',
                   linewidth=0.25)
ax_cart.plot(manifold_cartesian_points_amb.grid[0][2],
             manifold_cartesian_points_amb.grid[1][2],
             color='black')
ax_cart.plot(manifold_cartesian_points_amb.grid[0].T[2],
             manifold_cartesian_points_amb.grid[1].T[2],
             color='black')
ax_cart.set_xlim(-2, 4)
ax_cart.set_ylim(-3, 3)
ax_cart.set_xticks([])
ax_cart.set_yticks([])
ax_cart.set_aspect('equal')
ax_cart.set_axisbelow(True)



###
# Plot the Cartesian representation of the rectangle as it appears in the Cartesian chart
ax_cart_chart = plt.subplot(3, 4, 9)
ax_cart_chart.plot(*q_set_cartesian.grid, color=spot_color)
ax_cart_chart.pcolormesh(*manifold_cartesian_points.grid,
                   np.zeros([manifold_cartesian_points.shape[0] - 1, manifold_cartesian_points.shape[1] - 1]),
                   edgecolor='grey',
                   facecolor='none',
                   linewidth=0.25)
ax_cart_chart.plot(manifold_cartesian_points.grid[0][2],
             manifold_cartesian_points.grid[1][2],
             color='black')
ax_cart_chart.plot(manifold_cartesian_points.grid[0].T[2],
             manifold_cartesian_points.grid[1].T[2],
             color='black')
ax_cart_chart.set_xlim(-2, 2)
ax_cart_chart.set_ylim(-2, 2)
ax_cart_chart.set_xticks([-1, 0, 1, 2, 3, 4])
ax_cart_chart.set_yticks([-1, 0, 1, 2, 3, 4])
ax_cart_chart.set_aspect('equal')
ax_cart_chart.set_axisbelow(True)
ax_cart_chart.axhline(0, color='black', zorder=.75)
ax_cart_chart.axvline(0, color='black', zorder=.75)


###
# Plot the polar representation of the rectangle as it appears in the ambient space
ax_polar = plt.subplot(3, 4, 7)
ax_polar.plot(*q_set_ambient.grid, color=spot_color)
ax_polar.pcolormesh(*polar_grid_points_amb.grid,
                    np.zeros([polar_grid_points.shape[0] - 1, polar_grid_points.shape[1] - 1]),
                    edgecolor='grey',
                    facecolor='none',
                    linewidth=0.25)
ax_polar.set_aspect('equal')
ax_polar.plot(polar_grid_points_amb.grid[0][10][:],
              polar_grid_points_amb.grid[1][10][:],
              color='black')
ax_polar.set_xlim(-2, 4)
ax_polar.set_ylim(-3, 3)
ax_polar.set_xticks([])
ax_polar.set_yticks([])


ax_polar_chart = plt.subplot(3, 4, 11)
ax_polar_chart.plot(*q_set_polar.grid, color=spot_color)
ax_polar_chart.pcolormesh(*polar_grid_points.grid,
                    np.zeros([polar_grid_points.shape[0] - 1, polar_grid_points.shape[1] - 1]),
                    edgecolor='grey',
                    facecolor='none',
                    linewidth=0.25)
ax_polar_chart.set_xticks([0, 1, 2, 3, 4])
ax_polar_chart.set_yticks([-4, -2, 0, 2, 4])
ax_polar_chart.axhline(0, color='black', zorder=.75)
ax_polar_chart.axvline(0, color='black', zorder=.75)
ax_polar_chart.set_axisbelow(True)
ax_polar_chart.set_xlim(0, 4)
ax_polar_chart.set_ylim(-3, 3)

plt.show()


# ###
# # Directly embed the polar grid in the ambient space
# rg, thetag = embed_map_rt(polar_grid_manifold_elements).grid
#
# ax_polar2 = plt.subplot(3, 4, 8)
# ax_polar2.plot(q_set_ambient.grid[0], q_set_ambient.grid[1], color='grey')
# ax_polar2.pcolormesh(rg, thetag, np.zeros([rg.shape[0] - 1, rg.shape[1] - 1]), edgecolor=spot_color, facecolor='none',
#                      linewidth=0.25)
# ax_polar2.set_aspect('equal')
# ax_polar2.plot(xg[2][2:], yg[2][2:], color='black')
# #ax_polar.plot(xg.T[2][1:5], yg.T[2][1:5], color='black')
# ax_polar2.set_xlim(-2, 4)
# ax_polar2.set_ylim(-3, 3)
# ax_polar2.set_xticks([])  # [-1, 0, 1, 2, 3, 4])
# ax_polar2.set_yticks([])  # [-1, 0, 1, 2, 3, 4])
#
# ###
# # Directly parameterize the rectangle into the polar chart
# q_set_polar_param = param_map_rt(q_set_ambient)
#
# ax_polar_chart2 = plt.subplot(3, 4, 12)
# ax_polar_chart2.plot(q_set_polar.grid[0], q_set_polar.grid[1], color='grey')
# ax_polar_chart2.set_xlim(0, 4)
# ax_polar_chart2.set_ylim(-3, 3)
# ax_polar_chart2.set_xticks([0, 1, 2, 3, 4])
# ax_polar_chart2.set_yticks([-4, -2, 0, 2, 4])
# ax_polar_chart2.axhline(0, color='black', zorder=.75)
# ax_polar_chart2.axvline(0, color='black', zorder=.75)
# ax_polar_chart2.set_axisbelow(True)
# ax_polar_chart2.grid(True)
#
# plt.show()
