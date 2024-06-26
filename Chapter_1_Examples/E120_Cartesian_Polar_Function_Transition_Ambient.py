import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from geomotion import manifold as md
from geomotion import utilityfunctions as ut
from geomotion import plottingfunctions as gplt

spot_color = gplt.crimson
cmp = gplt.crimson_cmp


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
    theta = np.pi / 3
    rotmatrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    cartesian_coords = np.squeeze(np.matmul(rotmatrix, ambient_coords[:, None]))
    return cartesian_coords


def cartesian_to_ambient(ambient_coords):
    theta = -np.pi / 3
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
# For plotting purposes, create a rotated cartesian grid and a polar grid

# Generate a densely spaced set of points on two axes
x = np.linspace(-10, 10)
y = np.linspace(-10, 10)

# Generate a loosely spaced set of points on two axes
x_l = np.linspace(-10, 10, 7)
y_l = np.linspace(-10, 10, 7)

# Build meshgrids from x and y, using the meshgrid_array function, which generates a GridArray with the
# n_outer value generated automatically
cart_grid_points = ut.meshgrid_array(x, y)
cart_grid_points_l = ut.meshgrid_array(x_l, y_l)

# Turn the meshgrids into  ManifoldElementSets, specified in the Cartesian chart
cart_grid_manifold_elements = md.ManifoldElementSet(R2, cart_grid_points, 0)
cart_grid_manifold_elements_l = md.ManifoldElementSet(R2, cart_grid_points_l, 0)

# Convert (i.e. rotate) the grid into the ambient-space coordinates
cart_grid_rotated_manifold_elements = cart_grid_manifold_elements.transition(2)
cart_grid_rotated_manifold_elements_l = cart_grid_manifold_elements_l.transition(2)

# Get the grid representation of the ambient-space loose Cartesian meshgrid points
cart_grid_rotated_points = cart_grid_rotated_manifold_elements.grid
cart_grid_rotated_points_l = cart_grid_rotated_manifold_elements_l.grid

# Extract the x and y components of the grids
xg_ambient = cart_grid_points[0]
yg_ambient = cart_grid_points[1]

xg_ambient_l = cart_grid_points_l[0]
yg_ambient_l = cart_grid_points_l[1]

xg_cart = cart_grid_rotated_points[0]
yg_cart = cart_grid_rotated_points[1]

xg_cart_l = cart_grid_rotated_points_l[0]
yg_cart_l = cart_grid_rotated_points_l[1]

## Polar grid
# Generate a densely-spaced set of points for polar axes
r = np.linspace(.25, 10, 100)
theta = np.linspace(-3, 3, 100)

# Generate a loosely-spaced set of points for polar axes
r_l = np.linspace(.25, 10, 7)
theta_l = np.linspace(-3, 3, 21)

# Build meshgrids from r and theta, using the meshgrid_array function, which generates a GridArray with the
# n_outer value generated automatically
polar_grid_points = ut.meshgrid_array(r, theta)
polar_grid_points_l = ut.meshgrid_array(r_l, theta_l)

# Turn the meshgrids into ManifoldElementSets, specified in the polar chart
polar_grid_manifold_elements = md.ManifoldElementSet(R2, polar_grid_points, 1)
polar_grid_manifold_elements_l = md.ManifoldElementSet(R2, polar_grid_points_l, 1)

# Convert the grid into the ambient-space coordinates
polar_grid_cartesian_manifold_elements = polar_grid_manifold_elements.transition(0)
polar_grid_embedded_manifold_elements = polar_grid_cartesian_manifold_elements.transition(2)

polar_grid_cartesian_manifold_elements_l = polar_grid_manifold_elements_l.transition(0)
polar_grid_embedded_manifold_elements_l = polar_grid_cartesian_manifold_elements_l.transition(2)

# Get the grid representation of the ambient-space meshgrid points
polar_grid_embedded_points = polar_grid_embedded_manifold_elements.grid
polar_grid_embedded_points_l = polar_grid_embedded_manifold_elements_l.grid

# Extract the r and theta components of the embedded grid
rg = polar_grid_embedded_points[0]
thetag = polar_grid_embedded_points[1]

rg_l = polar_grid_embedded_points_l[0]
thetag_l = polar_grid_embedded_points_l[1]

##############
# Plot the calculated terms

# Use my red color for the plots
spot_color = gplt.crimson

###
# Plot the rectangle in the ambient space, with no coordinate grid (because the ambient space in principle is
# coordinate-free, and we've only supplied it with coordinates so we can actually tell the computer to plot things
ax_ambient = plt.subplot(3, 3, 2)
ax_ambient.pcolormesh(xg_ambient, yg_ambient, f_ambient.grid(cart_grid_points), cmap=cmp)
#ax_ambient.set_xlim(-2, 4)
#ax_ambient.set_ylim(-3, 3)
ax_ambient.set_xticks([])  # [-1, 0, 1, 2, 3, 4])
ax_ambient.set_yticks([])  # [-1, 0, 1, 2, 3, 4])
ax_ambient.set_aspect('equal')
ax_ambient.set_axisbelow(True)

# Plot the Cartesian grid and the function as they appear in ambient space
ax_cart = plt.subplot(3, 3, 4)
ax_cart.pcolormesh(xg_cart, yg_cart, f_xy.grid(cart_grid_points), cmap=cmp)
ax_cart.pcolormesh(xg_cart_l, yg_cart_l, np.zeros([xg_cart_l.shape[0] - 1, xg_cart_l.shape[1] - 1]), edgecolor='grey',
                   facecolor='none',
                   linewidth=0.25)
ax_cart.plot(xg_cart_l[3][2:], yg_cart_l[3][2:], color='black')
ax_cart.plot(xg_cart_l.T[3][2:], yg_cart_l.T[3][2:], color='black')
ax_cart.set_xlim(-10, 10)
ax_cart.set_ylim(-10, 10)
ax_cart.set_xticks([])
ax_cart.set_yticks([])
ax_cart.set_aspect('equal')
ax_cart.set_axisbelow(True)

###
# Plot the Cartesian representation of the function as it appears in the Cartesian chart
func_cart_chart = ma.masked_array(f_xy.grid(cart_grid_points),
                                  mask=((xg_cart > 10) | (yg_cart > 10) | (xg_cart < -10) | (yg_cart < -10)))

ax_cart_chart = plt.subplot(3, 3, 7)
ax_cart_chart.pcolormesh(xg_ambient, yg_ambient, func_cart_chart, cmap=cmp)
ax_cart_chart.pcolormesh(xg_ambient_l, yg_ambient_l, np.zeros([xg_ambient_l.shape[0] - 1, xg_ambient_l.shape[1] - 1]),
                         edgecolor='grey', facecolor='none',
                         linewidth=0.1)
# ax_cart_chart.set_xlim(-2, 2)
# ax_cart_chart.set_ylim(-2, 2)
# ax_cart_chart.set_xticks([-1, 0, 1, 2, 3, 4])
# ax_cart_chart.set_yticks([-1, 0, 1, 2, 3, 4])
ax_cart_chart.set_aspect('equal')
ax_cart_chart.set_axisbelow(True)
ax_cart_chart.grid(True)
ax_cart_chart.axhline(0, color='black', zorder=.75)
ax_cart_chart.axvline(0, color='black', zorder=.75)

ax_cart_chart.plot(xg_ambient_l[3][2:], yg_ambient_l[3][2:], color='black')
ax_cart_chart.plot(xg_ambient_l.T[3][2:], yg_ambient_l.T[3][2:], color='black')

###
# Plot the polar representation of the function as it appears in the ambient space
func_polar_ambient = f_rt.grid(polar_grid_points)

ax_polar = plt.subplot(3, 3, 6)
ax_polar.pcolormesh(rg, thetag, func_polar_ambient, cmap=cmp)
ax_polar.pcolormesh(rg_l, thetag_l, np.zeros([rg_l.shape[0] - 1, rg_l.shape[1] - 1]), edgecolor='grey',
                    facecolor='none', linewidth=0.1)
ax_polar.set_aspect('equal')
ax_polar.plot(xg_cart_l[3][3:], yg_cart_l[3][3:], color='black')

ax_polar_chart = plt.subplot(3, 3, 9)
ax_polar_chart.pcolormesh(polar_grid_points[0], polar_grid_points[1], func_polar_ambient, cmap=cmp)
ax_polar_chart.pcolormesh(polar_grid_points_l[0], polar_grid_points_l[1],
                          np.zeros([polar_grid_points_l[0].shape[0] - 1, polar_grid_points_l[0].shape[1] - 1]),
                          edgecolor='grey', facecolor='none',
                          linewidth=0.1)

ax_polar_chart.set_aspect('equal')
ax_polar_chart.set_axisbelow(True)
ax_polar_chart.axhline(0, color='black', zorder=2)
ax_polar_chart.axvline(0, color='black', zorder=.75)

plt.show()
