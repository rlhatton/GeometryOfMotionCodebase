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
                      l* np.sin(2 * np.pi * chart_coords[1])]

    return ambient_coords


def second_embedding(chart_coords):
    chart_coords[1] = np.mod(chart_coords[1] + 0.25, 1)
    ambient_coords = [np.cos(2 * np.pi * chart_coords[0]) * (L + (l * np.cos(2 * np.pi * chart_coords[1]))),
                      np.sin(2 * np.pi * chart_coords[0]) * (L + (l * np.cos(2 * np.pi * chart_coords[1]))),
                      l* np.sin(2 * np.pi * chart_coords[1])]

    return ambient_coords

def third_embedding(chart_coords):
    chart_coords[0] = np.mod(chart_coords[0] + 0.25, 1)
    chart_coords[1] = np.mod(chart_coords[1] - 0.25, 1)
    ambient_coords = [np.cos(2 * np.pi * chart_coords[0]) * (L + (l * np.cos(2 * np.pi * chart_coords[1]))),
                      np.sin(2 * np.pi * chart_coords[0]) * (L + (l * np.cos(2 * np.pi * chart_coords[1]))),
                      l* np.sin(2 * np.pi * chart_coords[1])]

    return ambient_coords


embed_first = md.ManifoldMap(Q, Q_amb, first_embedding)
embed_second = md.ManifoldMap(Q, Q_amb, second_embedding)
embed_third = md.ManifoldMap(Q, Q_amb, third_embedding)

# Make a grid of coordinate values
coordinate_grid = ut.meshgrid_array(np.linspace(-.4, .4, 25), np.linspace(-.4, .4, 25))

# Embed coordinate grids for three two charts into R3
first_grid_Q_amb = embed_first(coordinate_grid).grid
second_grid_Q_amb = embed_second(coordinate_grid).grid
third_grid_Q_amb = embed_third(coordinate_grid).grid



fig = plt.figure()


# Note that composite image renders poorly because matplotlib doesn't really do 3d
ax_ambient0 = fig.add_subplot(332, projection='3d')
ax = ax_ambient0
ax.plot_surface(first_grid_Q_amb[0], first_grid_Q_amb[1], first_grid_Q_amb[2], facecolor='white', edgecolor='black',
                alpha=0.6)
ax.plot_surface(second_grid_Q_amb[0] * .9, second_grid_Q_amb[1] * .9, second_grid_Q_amb[2] * .9 , facecolor='white',
                edgecolor=spot_color, alpha=0.6)
ax.plot_surface(third_grid_Q_amb[0] * 1.1, third_grid_Q_amb[1] * 1.1, third_grid_Q_amb[2] * 1.1 , facecolor='white',
                edgecolor='grey', alpha=0.6)
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_aspect('equal')

ax_ambient1 = fig.add_subplot(334, projection='3d')
ax = ax_ambient1
ax.plot_surface(first_grid_Q_amb[0], first_grid_Q_amb[1], first_grid_Q_amb[2], facecolor='white', edgecolor=spot_color)
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_aspect('equal')

ax_ambient2 = fig.add_subplot(335, projection='3d')
ax = ax_ambient2
ax.plot_surface(second_grid_Q_amb[0], second_grid_Q_amb[1], second_grid_Q_amb[2], facecolor='white', edgecolor='black')
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_aspect('equal')

ax_ambient3 = fig.add_subplot(336, projection='3d')
ax = ax_ambient3
ax.plot_surface(third_grid_Q_amb[0], third_grid_Q_amb[1], third_grid_Q_amb[2], facecolor='white', edgecolor='grey')
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_aspect('equal')



plt.show()
