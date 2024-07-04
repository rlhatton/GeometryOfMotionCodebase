import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from geomotion import manifold as md
from geomotion import utilityfunctions as ut
from geomotion import plottingfunctions as gplt
from S100_Construct_R2 import R2

spot_color = gplt.crimson
cmp = gplt.crimson_cmp


##########
# Set up the transition map for R2 with the standard polar and Cartesian charts and a third chart that represents the
# "ambient" or "underlying" Euclidean space over which the charts are defined


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

param_map_xy = md.ManifoldMap(Q_amb, Q, ambient_to_cartesian, 0, 0)
embed_map_xy = md.ManifoldMap(Q, Q_amb, cartesian_to_ambient, 0, 0)

param_map_rt = param_map_xy.transition_output(1)
embed_map_rt = embed_map_xy.transition(1)


#####################
# Construct the function in polar coordinates
def monkey_saddle(coords):
    f = coords[0] * coords[0] * coords[0] * np.cos(3 * coords[1])
    return f


# Make the monkey saddle function into a manifold function
f_rt = md.ManifoldFunction(Q, monkey_saddle, 1)

# Convert the function to Cartesian and ambient coordinates
f_xy = f_rt.transition(0)
f_ambient = f_rt.pullback(param_map_rt)
f_xy_double_pullback = f_ambient.pullback(embed_map_xy)

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
cart_grid_rotated_manifold_elements = embed_map_xy(cart_grid_manifold_elements)
cart_grid_rotated_manifold_elements_l = embed_map_xy(cart_grid_manifold_elements_l)

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

# Polar grid
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
polar_grid_embedded_manifold_elements = embed_map_rt(polar_grid_manifold_elements)

polar_grid_embedded_manifold_elements_l = embed_map_rt(polar_grid_manifold_elements_l)

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

###
# Plot the function in the ambient space, with no coordinate grid (because the ambient space in principle is
# coordinate-free, and we've only supplied it with coordinates so we can actually tell the computer to plot things
ax_ambient = plt.subplot(3, 4, 3)
ax_ambient.pcolormesh(xg_ambient, yg_ambient, f_ambient(cart_grid_points), cmap=cmp)
# ax_ambient.set_xlim(-2, 4)
# ax_ambient.set_ylim(-3, 3)
ax_ambient.set_xticks([])  # [-1, 0, 1, 2, 3, 4])
ax_ambient.set_yticks([])  # [-1, 0, 1, 2, 3, 4])
ax_ambient.set_aspect('equal')
ax_ambient.set_axisbelow(True)

# Plot the Cartesian grid and the function as they appear in ambient space
ax_cart = plt.subplot(3, 4, 6)
ax_cart.pcolormesh(xg_cart, yg_cart, f_xy(cart_grid_points), cmap=cmp)
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

# Plot the Cartesian grid and the doubly-pulledback function as they appear in ambient space
ax_cart = plt.subplot(3, 4, 5)
ax_cart.pcolormesh(xg_cart, yg_cart, f_xy_double_pullback(cart_grid_points), cmap=cmp)
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
func_cart_chart = ma.masked_array(f_xy(cart_grid_points),
                                  mask=((xg_cart > 10) | (yg_cart > 10) | (xg_cart < -10) | (yg_cart < -10)))

ax_cart_chart = plt.subplot(3, 4, 10)
ax_cart_chart.pcolormesh(xg_ambient, yg_ambient, func_cart_chart, cmap=cmp)
ax_cart_chart.pcolormesh(xg_ambient_l, yg_ambient_l, np.zeros([xg_ambient_l.shape[0] - 1, xg_ambient_l.shape[1] - 1]),
                         edgecolor='grey', facecolor='none',
                         linewidth=0.1)

ax_cart_chart.set_aspect('equal')
ax_cart_chart.set_axisbelow(True)
ax_cart_chart.grid(True)
ax_cart_chart.axhline(0, color='black', zorder=.75)
ax_cart_chart.axvline(0, color='black', zorder=.75)

ax_cart_chart.plot(xg_ambient_l[3][2:], yg_ambient_l[3][2:], color='black')
ax_cart_chart.plot(xg_ambient_l.T[3][2:], yg_ambient_l.T[3][2:], color='black')

# Plot the Cartesian representation of the doubly-pulledback function as it appears in the Cartesian chart
func_cart_chart = ma.masked_array(f_xy_double_pullback(cart_grid_points),
                                  mask=((xg_cart > 10) | (yg_cart > 10) | (xg_cart < -10) | (yg_cart < -10)))

ax_cart_chart = plt.subplot(3, 4, 9)
ax_cart_chart.pcolormesh(xg_ambient, yg_ambient, func_cart_chart, cmap=cmp)
ax_cart_chart.pcolormesh(xg_ambient_l, yg_ambient_l, np.zeros([xg_ambient_l.shape[0] - 1, xg_ambient_l.shape[1] - 1]),
                         edgecolor='grey', facecolor='none',
                         linewidth=0.1)

ax_cart_chart.set_aspect('equal')
ax_cart_chart.set_axisbelow(True)
ax_cart_chart.grid(True)
ax_cart_chart.axhline(0, color='black', zorder=.75)
ax_cart_chart.axvline(0, color='black', zorder=.75)

ax_cart_chart.plot(xg_ambient_l[3][2:], yg_ambient_l[3][2:], color='black')
ax_cart_chart.plot(xg_ambient_l.T[3][2:], yg_ambient_l.T[3][2:], color='black')

###
# Plot the polar representation of the function as it appears in the ambient space
func_polar_ambient = f_rt(polar_grid_points)

ax_polar = plt.subplot(3, 4, 8)
ax_polar.pcolormesh(rg, thetag, func_polar_ambient, cmap=cmp)
ax_polar.pcolormesh(rg_l, thetag_l, np.zeros([rg_l.shape[0] - 1, rg_l.shape[1] - 1]), edgecolor='grey',
                    facecolor='none', linewidth=0.1)
ax_polar.set_aspect('equal')
ax_polar.plot(xg_cart_l[3][3:], yg_cart_l[3][3:], color='black')
ax_polar.set_xticks([])
ax_polar.set_yticks([])

ax_polar_chart = plt.subplot(3, 4, 12)
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
