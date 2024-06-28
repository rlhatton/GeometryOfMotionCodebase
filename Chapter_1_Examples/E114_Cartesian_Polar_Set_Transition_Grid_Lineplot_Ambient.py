import numpy as np
import matplotlib.pyplot as plt
from geomotion import manifold as md
from geomotion import utilityfunctions as ut
from geomotion import plottingfunctions as gplt



##########
# Set up the transition map for R2 with the standard polar and Cartesian charts and a third chart that represents the
# "ambient" or "underlying" Euclidean space over which the charts are defined

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
    theta = np.pi/3
    rotmatrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    cartesian_coords = np.squeeze(np.matmul(rotmatrix, ambient_coords[:, None]))
    return cartesian_coords

def cartesian_to_ambient(ambient_coords):
    theta = -np.pi/3
    rotmatrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    cartesian_coords = np.squeeze(np.matmul(rotmatrix, ambient_coords[:, None]))
    return cartesian_coords

# Place the transition maps into the transition table
transition_table = [[None, cartesian_to_polar, cartesian_to_ambient], [polar_to_cartesian, None, None], [ambient_to_cartesian, None, None]]

# Construct the R2 manifold from the transition table
R2 = md.Manifold(transition_table, 2)

# Make the working manifold for this problem R2
Q = R2



#####################
# Construct the points describing a rotated rectangle in the ambient space

# Specify the rectangle's geometry in the ambient-space chart
box_width = 1
box_height = 3
box_angle = -np.pi/3
box_offset_x = 2
box_offset_y = .5

# Build the edges of the box counterclockwise from the lower-right corner
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

# Combine the x and y edges into a single 2xN array
edge = np.stack([edge_x, edge_y])

# Construct a rotation matrix corresponding to the box angle
rotmatrix = np.array([[np.cos(box_angle), -np.sin(box_angle)], [np.sin(box_angle), np.cos(box_angle)]])

# Rotate the box and apply the offset
edge = np.matmul(rotmatrix,edge) + [[box_offset_x], [box_offset_y]]

########
# Collect these points into a ManifoldElementSet
q_numeric = ut.GridArray(edge, 1)  # This specifies that the first dimension is the coordinate values, and the remaining dimensions are for different points
q_set_ambient = md.ManifoldElementSet(Q, q_numeric, 2) #  Turn the data into ManifoldElements as specified in the ambient (third) chart

# Convert the rectangle into a set of points specified in the Cartesian chart
q_set_cartesian = q_set_ambient.transition(0)

# Convert the rectangle into a set of points specified in the polar chart
q_set_polar = q_set_cartesian.transition(1)


############
# For plotting purposes, create a rotated cartesian grid and a polar grid

# Generate a loosely spaced set of points on two axes
x = np.linspace(-2, 4, 7)
y = np.linspace(-2, 4, 7)

# Build a meshgrid from x and y, using the meshgrid_array function, which generates a GridArray with the
# n_outer value generated automatically
cart_grid_points = ut.meshgrid_array(x,y)

# Turn the meshgrid into a ManifoldElementSet, specified in the Cartesian chart
cart_grid_manifold_elements = md.ManifoldElementSet(R2, cart_grid_points, 0)

# Convert (i.e. rotate) the grid into the ambient-space coordinates
cart_grid_rotated_manifold_elements = cart_grid_manifold_elements.transition(2)

# Get the grid representation of the ambient-space meshgrid points
cart_grid_rotated_points = cart_grid_rotated_manifold_elements.grid

# Extract the x and y components of the rotated grid
xg = cart_grid_rotated_points[0]
yg = cart_grid_rotated_points[1]

##
# Generate a loosely-spaced set of points for polar axes
r = np.linspace(.25, 4, 7)
theta = np.linspace(-3, 3, 21)

# Build a meshgrid from r and theta, using the meshgrid_array function, which generates a GridArray with the
# n_outer value generated automatically
polar_grid_points = ut.meshgrid_array(r, theta)

# Turn the meshgrid into a ManifoldElementSet, specified in the polar chart
polar_grid_manifold_elements = md.ManifoldElementSet(R2, polar_grid_points, 1)

# Convert the grid into the ambient-space coordinates
polar_grid_cartesian_manifold_elements = polar_grid_manifold_elements.transition(0)
polar_grid_embedded_manifold_elements = polar_grid_cartesian_manifold_elements.transition(2)

# Get the grid representation of the ambient-space meshgrid points
polar_grid_embedded_points = polar_grid_embedded_manifold_elements.grid

# Extract the r and theta components of the embedded grid
rg = polar_grid_embedded_points[0]
thetag = polar_grid_embedded_points[1]

##############
# Plot the calculated terms

# Use my red color for the plots
spot_color = gplt.crimson

###
# Plot the rectangle in the ambient space, with no coordinate grid (because the ambient space in principle is
# coordinate-free, and we've only supplied it with coordinates so we can actually tell the computer to plot things
ax_ambient = plt.subplot(3, 3, 2)
ax_ambient.plot(q_set_ambient.grid[0], q_set_ambient.grid[1], color=spot_color)
ax_ambient.set_xlim(-2, 4)
ax_ambient.set_ylim(-3, 3)
ax_ambient.set_xticks([])  # [-1, 0, 1, 2, 3, 4])
ax_ambient.set_yticks([])  # [-1, 0, 1, 2, 3, 4])
ax_ambient.set_aspect('equal')
ax_ambient.set_axisbelow(True)


# Plot the Cartesian grid and the rectangle as they appear in ambient space
ax_cart = plt.subplot(3, 3, 4)
ax_cart.plot(q_set_ambient.grid[0], q_set_ambient.grid[1], color=spot_color)
ax_cart.pcolormesh(xg, yg, np.zeros([xg.shape[0]-1, xg.shape[1]-1]), edgecolor='grey', facecolor='none', linewidth=0.25)
ax_cart.plot(xg[2], yg[2], color='black')
ax_cart.plot(xg.T[2], yg.T[2], color='black')
ax_cart.set_xlim(-2, 4)
ax_cart.set_ylim(-3, 3)
ax_cart.set_xticks([])
ax_cart.set_yticks([])
ax_cart.set_aspect('equal')
ax_cart.set_axisbelow(True)


###
# Plot the Cartesian representation of the rectangle as it appears in the Cartesian chart
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


###
# Plot the polar representation of the rectangle as it appears in the ambient space
ax_polar = plt.subplot(3, 3, 6)
ax_polar.plot(q_set_ambient.grid[0], q_set_ambient.grid[1], color=spot_color)
ax_polar.pcolormesh(rg, thetag, np.zeros([rg.shape[0]-1, rg.shape[1]-1]), edgecolor='grey', facecolor='none', linewidth=0.25)
ax_polar.set_aspect('equal')
ax_polar.plot(xg[2][2:], yg[2][2:], color='black')
#ax_polar.plot(xg.T[2][1:5], yg.T[2][1:5], color='black')
ax_polar.set_xlim(-2, 4)
ax_polar.set_ylim(-3, 3)
ax_polar.set_xticks([])  # [-1, 0, 1, 2, 3, 4])
ax_polar.set_yticks([])  # [-1, 0, 1, 2, 3, 4])

ax_polar_chart = plt.subplot(3, 3, 9)
ax_polar_chart.plot(q_set_polar.grid[0], q_set_polar.grid[1], color=spot_color)
ax_polar_chart.set_xlim(0, 4)
ax_polar_chart.set_ylim(-3, 3)
ax_polar_chart.set_xticks([0, 1, 2, 3, 4])
ax_polar_chart.set_yticks([-4, -2, 0, 2, 4])
ax_polar_chart.axhline(0, color='black', zorder=.75)
ax_polar_chart.axvline(0, color='black', zorder=.75)
ax_polar_chart.set_axisbelow(True)
ax_polar_chart.grid(True)

plt.show()