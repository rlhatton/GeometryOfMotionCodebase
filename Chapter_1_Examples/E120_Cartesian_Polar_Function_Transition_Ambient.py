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
    theta = np.pi / 6
    rotmatrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    cartesian_coords = np.squeeze(np.matmul(rotmatrix, ambient_coords[:, None]))
    return cartesian_coords


def cartesian_to_ambient(ambient_coords):
    theta = -np.pi / 6
    rotmatrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    cartesian_coords = np.squeeze(np.matmul(rotmatrix, ambient_coords[:, None]))
    return cartesian_coords


# Place the transition maps into the transition table
transition_table = [[None, cartesian_to_polar, cartesian_to_ambient], [polar_to_cartesian, None, None],
                    [ambient_to_cartesian, None, None]]

# Construct the R2 manifold from the transition table
R2 = md.Manifold(transition_table, 2)

# Make the working manifold for this problem R2
Q = R2


#####################
# Construct the function in the polar coordinates
def monkey_saddle(coords):
    f = coords[0] * coords[0] * coords[0] * np.cos(3 * coords[1])
    return f


f_rt = md.ManifoldFunction(Q, monkey_saddle, 1)

# Convert the function to Cartesian and ambient coordinates
f_xy = f_rt.transition(0)
f_ambient = f_xy.transition(2)



############
# For plotting purposes, create a rotated cartesian grid

# Generate a densely spaced set of points on two axes
x = np.linspace(-10, 10)
y = np.linspace(-10, 10)

# Build a meshgrid from x and y, using the meshgrid_array function, which generates a GridArray with the
# n_outer value generated automatically
cart_grid_points = ut.meshgrid_array(x, y)

# Turn the meshgrid into a ManifoldElementSet, specified in the Cartesian chart
cart_grid_manifold_elements = md.ManifoldElementSet(R2, cart_grid_points, 0)

# Convert (i.e. rotate) the grid into the ambient-space coordinates
cart_grid_rotated_manifold_elements = cart_grid_manifold_elements.transition(2)

# Get the grid representation of the ambient-space meshgrid points
cart_grid_rotated_points = cart_grid_rotated_manifold_elements.grid

# Extract the x and y components of the grids
xg_ambient = cart_grid_points[0]
yg_ambient = cart_grid_points[1]

xg_cart = cart_grid_rotated_points[0]
yg_cart = cart_grid_rotated_points[1]

##############
# Plot the calculated terms

# Use my red color for the plots
spot_color = gplt.crimson

###
# Plot the rectangle in the ambient space, with no coordinate grid (because the ambient space in principle is
# coordinate-free, and we've only supplied it with coordinates so we can actually tell the computer to plot things
ax_ambient = plt.subplot(3, 3, 2)
ax_ambient.pcolormesh(xg_ambient, yg_ambient, f_ambient.grid(cart_grid_points))
#ax_ambient.set_xlim(-2, 4)
#ax_ambient.set_ylim(-3, 3)
ax_ambient.set_xticks([])  # [-1, 0, 1, 2, 3, 4])
ax_ambient.set_yticks([])  # [-1, 0, 1, 2, 3, 4])
ax_ambient.set_aspect('equal')
ax_ambient.set_axisbelow(True)


# Plot the Cartesian grid and the rectangle as they appear in ambient space
ax_cart = plt.subplot(3, 3, 4)
ax_cart.pcolormesh(xg_cart, yg_cart, f_xy(cart_grid_points))
ax_cart.pcolormesh(xg_cart, yg_cart, np.zeros([xg_cart.shape[0] - 1, xg_cart.shape[1] - 1]), edgecolor='grey', facecolor='none',
                   linewidth=0.25)
ax_cart.plot(xg_cart[2], yg_cart[2], color='black')
ax_cart.plot(xg_cart.T[2], yg_cart.T[2], color='black')
#ax_cart.set_xlim(-2, 4)
#ax_cart.set_ylim(-3, 3)
ax_cart.set_xticks([])
ax_cart.set_yticks([])
ax_cart.set_aspect('equal')
ax_cart.set_axisbelow(True)

###
# Plot the Cartesian representation of the rectangle as it appears in the Cartesian chart
ax_cart_chart = plt.subplot(3, 3, 7)
ax_cart_chart.pcolormesh(xg_ambient, yg_ambient, f_xy(cart_grid_points))
# ax_cart_chart.set_xlim(-2, 2)
# ax_cart_chart.set_ylim(-2, 2)
# ax_cart_chart.set_xticks([-1, 0, 1, 2, 3, 4])
# ax_cart_chart.set_yticks([-1, 0, 1, 2, 3, 4])
ax_cart_chart.set_aspect('equal')
ax_cart_chart.set_axisbelow(True)
ax_cart_chart.grid(True)
ax_cart_chart.axhline(0, color='black', zorder=.75)
ax_cart_chart.axvline(0, color='black', zorder=.75)

plt.show()

###
# Plot the polar representation of the rectangle as it appears in the Cartesian chart
ax_polar = plt.subplot(3, 3, 6, projection='polar')
ax_polar.plot(q_set_polar.grid[1], q_set_polar.grid[0], color=spot_color)
ax_polar.set_rlim(0, 6)
ut.convert_polar_xticks_to_radians(ax_polar)

ax_polar_chart = plt.subplot(3, 3, 9)
ax_polar_chart.plot(q_set_polar.grid[1], q_set_polar.grid[0], color=spot_color)

plt.show()
