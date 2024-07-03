import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from geomotion import manifold as md
from geomotion import utilityfunctions as ut
from geomotion import plottingfunctions as gplt
from S110_Construct_T2 import T2

spot_color = gplt.crimson

# Make the working manifold R1S1
Q = T2

# Make an R3 space with a single chart
Q_amb = md.Manifold([[None]], 3)

# Create embeddings for the torus charts, using unit radius in the ambient space
# for the major axis, and half-unit radius for the minor axis

L = 1
l = 0.5


def first_embedding(chart_coords):
    ambient_coords = [np.cos(2 * np.pi * chart_coords[0]) * (L + (l * np.cos(2 * np.pi * chart_coords[1]))),
                      np.sin(2 * np.pi * chart_coords[0]) * (L + (l * np.cos(2 * np.pi * chart_coords[1]))),
                      l * np.sin(2 * np.pi * chart_coords[1])]

    return ambient_coords


def second_embedding(chart_coords):
    chart_coords[0] = ut.cmod(chart_coords[0] - 0.5, 1)
    chart_coords[1] = ut.cmod(chart_coords[1] - 0.5, 1)
    ambient_coords = [np.cos(2 * np.pi * chart_coords[0]) * (L + (l * np.cos(2 * np.pi * chart_coords[1]))),
                      np.sin(2 * np.pi * chart_coords[0]) * (L + (l * np.cos(2 * np.pi * chart_coords[1]))),
                      l * np.sin(2 * np.pi * chart_coords[1])]

    return ambient_coords


def third_embedding(chart_coords):
    chart_coords[0] = ut.cmod(chart_coords[0] - 0.25, 1)
    chart_coords[1] = ut.cmod(chart_coords[1] - 0.25, 1)
    ambient_coords = [np.cos(2 * np.pi * chart_coords[0]) * (L + (l * np.cos(2 * np.pi * chart_coords[1]))),
                      np.sin(2 * np.pi * chart_coords[0]) * (L + (l * np.cos(2 * np.pi * chart_coords[1]))),
                      l * np.sin(2 * np.pi * chart_coords[1])]

    return ambient_coords


# Generate the embedding functions corresponding to the charts
embed_first = md.ManifoldMap(Q, Q_amb, first_embedding)
embed_second = md.ManifoldMap(Q, Q_amb, second_embedding)
embed_third = md.ManifoldMap(Q, Q_amb, third_embedding)

# Make a grid of coordinate values
coordinate_grid = ut.meshgrid_array(np.linspace(-.4, .4, 25), np.linspace(-.4, .4, 25))

# Create sets of configurations for the coordinate grids in each chart
first_points = md.ManifoldElementSet(Q, coordinate_grid, 0)
second_points = md.ManifoldElementSet(Q, coordinate_grid, 1)
third_points = md.ManifoldElementSet(Q, coordinate_grid, 2)

# Embed coordinate grids for three two charts into R3
first_grid_Q_amb = embed_first(first_points).grid
second_grid_Q_amb = embed_second(second_points).grid
third_grid_Q_amb = embed_third(third_points).grid

# Pull back the points defined in the second and third charts into the first chart
first_chart_second_grid = second_points.transition(0).grid
first_chart_third_grid = third_points.transition(0).grid

# Pull back the points defined in the first and third charts into the second chart
second_chart_first_grid = first_points.transition(1).grid
second_chart_third_grid = third_points.transition(1).grid

# Pull back the points defined in the first and second charts into the third chart
third_chart_first_grid = first_points.transition(2).grid
third_chart_second_grid = second_points.transition(2).grid

fig = plt.figure()


# * syntax unpacks the grid values along the first axis, so I don't need to write out each index
ax_ambient1 = fig.add_subplot(331, projection='3d')
ax = ax_ambient1
ax.plot_surface(*first_grid_Q_amb, facecolor='white', edgecolor=spot_color)
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_aspect('equal')

ax_ambient2 = fig.add_subplot(332, projection='3d')
ax = ax_ambient2
ax.plot_surface(*second_grid_Q_amb, facecolor='white', edgecolor='black')
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_aspect('equal')

ax_ambient3 = fig.add_subplot(333, projection='3d')
ax = ax_ambient3
ax.plot_surface(*third_grid_Q_amb, facecolor='white', edgecolor='grey')
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_aspect('equal')

ax_flat1 = fig.add_subplot(334)
ax = ax_flat1
ax.pcolormesh(*coordinate_grid, np.zeros([coordinate_grid.shape[1] - 1, coordinate_grid.shape[2] - 1]), facecolor='none', edgecolor=spot_color)
ax.scatter(*first_chart_second_grid, color='black', s=1) #
ax.grid(False)
# ax.set_xticks([])
# ax.set_yticks([])
ax.set_xlim(-.5, .5)
ax.set_ylim(-.5, .5)
ax.set_aspect('equal')

ax_flat2 = fig.add_subplot(337)
ax = ax_flat2
ax.pcolormesh(*coordinate_grid, np.zeros([coordinate_grid.shape[1] - 1, coordinate_grid.shape[2] - 1]), facecolor='none', edgecolor=spot_color)
ax.scatter(*first_chart_third_grid, color='grey', s=1) #
ax.grid(False)
# ax.set_xticks([])
# ax.set_yticks([])
ax.set_xlim(-.5, .5)
ax.set_ylim(-.5, .5)
ax.set_aspect('equal')


ax_flat3 = fig.add_subplot(335)
ax = ax_flat3
ax.pcolormesh(*coordinate_grid, np.zeros([coordinate_grid.shape[1] - 1, coordinate_grid.shape[2] - 1]), facecolor='none', edgecolor='black')
ax.scatter(*second_chart_first_grid, color=spot_color, s=1)
ax.grid(False)
# ax.set_xticks([])
# ax.set_yticks([])
ax.set_xlim(-.5, .5)
ax.set_ylim(-.5, .5)
ax.set_aspect('equal')

ax_flat4 = fig.add_subplot(338)
ax = ax_flat4
ax.pcolormesh(*coordinate_grid, np.zeros([coordinate_grid.shape[1] - 1, coordinate_grid.shape[2] - 1]), facecolor='none', edgecolor='black')
ax.scatter(*second_chart_third_grid, color='grey', s=1)
ax.grid(False)
# ax.set_xticks([])
# ax.set_yticks([])
ax.set_xlim(-.5, .5)
ax.set_ylim(-.5, .5)
ax.set_aspect('equal')

ax_flat5 = fig.add_subplot(336)
ax = ax_flat5
ax.pcolormesh(*coordinate_grid, np.zeros([coordinate_grid.shape[1] - 1, coordinate_grid.shape[2] - 1]), facecolor='none', edgecolor='grey')
ax.scatter(*third_chart_first_grid, color=spot_color, s=1) #
ax.grid(False)
# ax.set_xticks([])
# ax.set_yticks([])
ax.set_xlim(-.5, .5)
ax.set_ylim(-.5, .5)
ax.set_aspect('equal')

ax_flat6 = fig.add_subplot(339)
ax = ax_flat6
ax.pcolormesh(*coordinate_grid, np.zeros([coordinate_grid.shape[1] - 1, coordinate_grid.shape[2] - 1]), facecolor='none', edgecolor='grey')
ax.scatter(*third_chart_second_grid, color='black', s=1) #
ax.grid(False)
# ax.set_xticks([])
# ax.set_yticks([])
ax.set_xlim(-.5, .5)
ax.set_ylim(-.5, .5)
ax.set_aspect('equal')

fig = plt.figure()
ax_flat7 = fig.add_subplot(111)
ax = ax_flat7
ax.pcolormesh(*coordinate_grid, np.zeros([coordinate_grid.shape[1] - 1, coordinate_grid.shape[2] - 1]), facecolor='none', edgecolor='grey')
ax.scatter(*third_chart_second_grid, color='black', s=1)
ax.scatter(*third_chart_first_grid, color=spot_color, s=1)
ax.grid(False)
# ax.set_xticks([])
# ax.set_yticks([])
ax.set_xlim(-.5, .5)
ax.set_ylim(-.5, .5)
ax.set_aspect('equal')

plt.show()
