import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from geomotion import manifold as md
from geomotion import utilityfunctions as ut
from geomotion import plottingfunctions as gplt
from S110_Construct_R1S1 import R1S1

spot_color = gplt.crimson

# Make the working manifold R1S1
Q = R1S1

# Make an R3 space with a single chart
Q_amb = md.Manifold([[None]], 3)


# Create embeddings for the cylinder charts, using unit radius in the ambient space
# (note that this means the ambient and chart scales are different)

def front_embedding(front_coords):
    ambient_coords = [np.cos(2 * np.pi * front_coords[0]), np.sin(2 * np.pi * front_coords[0]),
                      2 * np.pi * front_coords[1]]

    return ambient_coords


def back_embedding(back_coords):
    ambient_coords = [-np.cos(2 * np.pi * back_coords[0]), np.sin(2 * np.pi * back_coords[0]),
                      2 * np.pi * back_coords[1]]

    return ambient_coords


cylinder_embedding = md.ManifoldMap(Q, Q_amb, [front_embedding, back_embedding], [0, 1])

# Make a grid of coordinate values
coordinate_grid = ut.meshgrid_array(np.linspace(-.4, .4, 25), np.linspace(-1, 1, 10))

# Instantiate sets of points defined as the generic coordinate grid evaluated in each chart
front_points = md.ManifoldElementSet(Q, coordinate_grid, 0)
back_points = md.ManifoldElementSet(Q, coordinate_grid, 1)

# Embed coordinate grids for the two charts into R3
front_Q_amb = cylinder_embedding(front_points)
back_Q_amb = cylinder_embedding(back_points)

fig = plt.figure()
ax_ambient1 = fig.add_subplot(131, projection='3d')
ax = ax_ambient1
ax.plot_surface(*front_Q_amb.grid, facecolor='white', edgecolor=spot_color)
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])

ax_ambient2 = fig.add_subplot(132, projection='3d')
ax = ax_ambient2
ax.plot_surface(*back_Q_amb.grid, facecolor='white', edgecolor='black')
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])

ax_ambient3 = fig.add_subplot(133, projection='3d')
ax = ax_ambient3
ax.plot_surface(*back_Q_amb.grid, facecolor='white', edgecolor='black', alpha=0.5)
ax.plot_surface(*(front_Q_amb.grid*.9), facecolor='white', edgecolor=spot_color, alpha=0.5)

ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])

plt.show()
