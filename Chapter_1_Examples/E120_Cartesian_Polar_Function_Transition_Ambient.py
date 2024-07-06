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

param_map = md.ManifoldMap(Q_amb, Q, ambient_to_cartesian, 0, 0)
embed_map = md.ManifoldMap(Q, Q_amb, cartesian_to_ambient, 0, 0)


#####################
# Construct the function in polar coordinates
def monkey_saddle(coords):
    f = coords[0] * coords[0] * coords[0] * np.cos(3 * coords[1])
    return f


# Make the monkey saddle function into a manifold function defined in the
# polar chart on the manifold
f_monkey_saddle = md.ManifoldFunction(Q, monkey_saddle, 1)

# pull back the function through the parameterization map to get an expression for
# the function on the ambient space
f_ambient = f_monkey_saddle.pullback(param_map)

# pull back the function through the embedding map to get an expression for the
# function defined in the Cartesian chart
f_double_pullback = f_ambient.pullback(embed_map)

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
cart_grid = ut.meshgrid_array(x, y)
cart_grid_l = ut.meshgrid_array(x_l, y_l)

# Turn the meshgrids into ManifoldElementSets, specified in the Cartesian chart on the ambient space
ambient_cartesian_points = md.ManifoldElementSet(Q_amb, cart_grid, 0)
ambient_cartesian_points_l = md.ManifoldElementSet(Q_amb, cart_grid_l, 0)

# Turn the meshgrids into  ManifoldElementSets, specified in the Cartesian chart on the manifold
manifold_cartesian_points = md.ManifoldElementSet(Q, cart_grid, 0)
manifold_cartesian_points_l = md.ManifoldElementSet(Q, cart_grid_l, 0)

# Convert (i.e. rotate) the grid of points on the manifold into the ambient-space coordinates
manifold_cartesian_points_amb = embed_map(manifold_cartesian_points)
manifold_cartesian_points_amb_l = embed_map(manifold_cartesian_points_l)

# Polar grid
# Generate a densely-spaced set of points for polar axes
r = np.linspace(.25, 10, 100)
theta = np.linspace(-3, 3, 100)

# Generate a loosely-spaced set of points for polar axes
r_l = np.linspace(.25, 10, 7)
theta_l = np.linspace(-3, 3, 21)

# Build meshgrids from r and theta, using the meshgrid_array function, which generates a GridArray with the
# n_outer value generated automatically
polar_grid = ut.meshgrid_array(r, theta)
polar_grid_l = ut.meshgrid_array(r_l, theta_l)

# Turn the meshgrids into ManifoldElementSets, specified in the polar chart
polar_grid_points = md.ManifoldElementSet(Q, polar_grid, 1)
polar_grid_points_l = md.ManifoldElementSet(Q, polar_grid_l, 1)

# Convert the polar grid into the ambient-space coordinates
polar_grid_points_amb = embed_map(polar_grid_points)

polar_grid_points_amb_l = embed_map(polar_grid_points_l)

##############
# Plot the calculated terms

###
# Plot the function in the ambient space, with no coordinate grid (because the ambient space in principle is
# coordinate-free, and we've only supplied it with coordinates so we can actually tell the computer to plot things
ax_ambient = plt.subplot(3, 4, 3)
ax_ambient.pcolormesh(*ambient_cartesian_points.grid, f_ambient(ambient_cartesian_points), cmap=cmp)
ax_ambient.set_aspect('equal')
ax_ambient.set_axisbelow(True)

# Plot the embedding of the Cartesian grid on the manifold and the ambient function evaluated over the
# ambient embedding of that grid
ax_cart = plt.subplot(3, 4, 6)
ax_cart.pcolormesh(*manifold_cartesian_points_amb.grid, f_ambient(manifold_cartesian_points_amb), cmap=cmp)
ax_cart.pcolormesh(*manifold_cartesian_points_amb_l.grid, np.zeros(
    [manifold_cartesian_points_amb_l.shape[0] - 1, manifold_cartesian_points_amb_l.shape[1] - 1]), edgecolor='grey',
                   facecolor='none',
                   linewidth=0.25)
ax_cart.plot(manifold_cartesian_points_amb_l.grid[0][3][2:], manifold_cartesian_points_amb_l.grid[1][3][2:],
             color='black')
ax_cart.plot(manifold_cartesian_points_amb_l.grid[0].T[3][2:], manifold_cartesian_points_amb_l.grid[1].T[3][2:],
             color='black')
ax_cart.set_xlim(-10, 10)
ax_cart.set_ylim(-10, 10)
ax_cart.set_xticks([])
ax_cart.set_yticks([])
ax_cart.set_aspect('equal')
ax_cart.set_axisbelow(True)

# Plot the Cartesian grid embedded into the ambient space, along with the value at each point on the grid of the
# function pulled back to the manifold
ax_cart = plt.subplot(3, 4, 5)
ax_cart.pcolormesh(*manifold_cartesian_points_amb.grid, f_double_pullback(manifold_cartesian_points), cmap=cmp)
ax_cart.pcolormesh(*manifold_cartesian_points_amb_l.grid, np.zeros(
    [manifold_cartesian_points_amb_l.shape[0] - 1, manifold_cartesian_points_amb_l.shape[1] - 1]), edgecolor='grey',
                   facecolor='none',
                   linewidth=0.25)
ax_cart.plot(manifold_cartesian_points_amb_l.grid[0][3][2:], manifold_cartesian_points_amb_l.grid[1][3][2:],
             color='black')
ax_cart.plot(manifold_cartesian_points_amb_l.grid[0].T[3][2:], manifold_cartesian_points_amb_l.grid[1].T[3][2:],
             color='black')
ax_cart.set_xlim(-10, 10)
ax_cart.set_ylim(-10, 10)
ax_cart.set_xticks([])
ax_cart.set_yticks([])
ax_cart.set_aspect('equal')
ax_cart.set_axisbelow(True)

###
# Plot the Cartesian representation of the function as it appears in the Cartesian chart, masking out any
# values for points that are outside the ambient-space bounds
func_cart_chart = ma.masked_array(f_monkey_saddle(manifold_cartesian_points),
                                  mask=((np.abs(manifold_cartesian_points_amb.grid[0]) > 10) | (
                                              np.abs(manifold_cartesian_points_amb.grid[1]) > 10)))

ax_cart_chart = plt.subplot(3, 4, 10)
ax_cart_chart.pcolormesh(*manifold_cartesian_points.grid, func_cart_chart, cmap=cmp)
ax_cart_chart.pcolormesh(*manifold_cartesian_points_l.grid,
                         np.zeros([manifold_cartesian_points_l.shape[0] - 1, manifold_cartesian_points_l.shape[1] - 1]),
                         edgecolor='grey', facecolor='none',
                         linewidth=0.1)

ax_cart_chart.set_aspect('equal')
ax_cart_chart.set_axisbelow(True)

ax_cart_chart.plot(manifold_cartesian_points_l.grid[0][3][2:], manifold_cartesian_points_l.grid[1][3][2:],
                   color='black')
ax_cart_chart.plot(manifold_cartesian_points_l.grid[0].T[3][2:], manifold_cartesian_points_l.grid[1].T[3][2:],
                   color='black')

# Plot the Cartesian representation of the doubly-pulledback function as it appears in the Cartesian chart
func_cart_chart = ma.masked_array(f_double_pullback(manifold_cartesian_points),
                                  mask=((np.abs(manifold_cartesian_points_amb.grid[0]) > 10) | (
                                              np.abs(manifold_cartesian_points_amb.grid[1]) > 10)))

ax_cart_chart = plt.subplot(3, 4, 9)
ax_cart_chart.pcolormesh(*manifold_cartesian_points.grid, func_cart_chart, cmap=cmp)
ax_cart_chart.pcolormesh(*manifold_cartesian_points_l.grid,
                         np.zeros([manifold_cartesian_points_l.shape[0] - 1, manifold_cartesian_points_l.shape[1] - 1]),
                         edgecolor='grey', facecolor='none',
                         linewidth=0.1)

ax_cart_chart.set_aspect('equal')
ax_cart_chart.set_axisbelow(True)

ax_cart_chart.plot(manifold_cartesian_points_l.grid[0][3][2:], manifold_cartesian_points_l.grid[1][3][2:],
                   color='black')
ax_cart_chart.plot(manifold_cartesian_points_l.grid[0].T[3][2:], manifold_cartesian_points_l.grid[1].T[3][2:],
                   color='black')

###
# Plot the polar representation of the function as it appears in the ambient space
ax_polar = plt.subplot(3, 4, 8)
ax_polar.pcolormesh(*polar_grid_points_amb.grid, f_ambient(polar_grid_points_amb), cmap=cmp)
ax_polar.pcolormesh(*polar_grid_points_amb_l.grid,
                    np.zeros([polar_grid_points_amb_l.shape[0] - 1, polar_grid_points_amb_l.shape[1] - 1]),
                    edgecolor='grey',
                    facecolor='none', linewidth=0.1)
ax_polar.set_aspect('equal')
ax_polar.plot(polar_grid_points_amb_l.grid[0][10][1:], polar_grid_points_amb_l.grid[1][10][1:], color='black')
ax_polar.set_xticks([])
ax_polar.set_yticks([])

ax_polar_chart = plt.subplot(3, 4, 12)
ax_polar_chart.pcolormesh(*polar_grid_points.grid, f_monkey_saddle(polar_grid_points), cmap=cmp)
ax_polar_chart.pcolormesh(*polar_grid_points_l.grid,
                          np.zeros([polar_grid_points_l.shape[0] - 1, polar_grid_points_l.shape[1] - 1]),
                          edgecolor='grey', facecolor='none',
                          linewidth=0.1)

ax_polar_chart.set_axisbelow(True)
ax_polar_chart.axhline(0, color='black', zorder=2)
ax_polar_chart.axvline(0, color='black', zorder=.75)

plt.show()
