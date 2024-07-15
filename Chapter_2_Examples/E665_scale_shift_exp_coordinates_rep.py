import numpy as np
from geomotion import manifold as md, diffmanifold as tb, utilityfunctions as ut, plottingfunctions as gplt, liegroup as lgp
from S550_Construct_RxRplus_rep import RxRplus
from matplotlib import pyplot as plt

# Make things print nicely
np.set_printoptions(precision=2)
spot_color = gplt.crimson

# Make the working group the scale-shift group
G = RxRplus

# Create a of exponential coordinate values
grid_xy = ut.meshgrid_array(np.linspace(-1, 1, 12), np.linspace(-1, 1, 12))

# Turn the grid into a set of points in the tangent space at the identity
e = G.identity_element()
g_box_points = RxRplus.vector_set(e, grid_xy)

# Exponentiate all of those vectors
g_points = g_box_points.exp_R

ax = plt.subplot(1, 1,1)
ax.pcolormesh(*g_points.grid,
                   np.zeros([g_points.shape[0] - 1, g_points.shape[1] - 1]),
                   edgecolor='grey',
                   facecolor='none',
                   linewidth=0.25)
ax.set_xlim(0, 3)
ax.set_ylim(-2, 2)
ax.set_aspect('equal')
ax.axhline(0, color='grey', linewidth=0.5, linestyle="dashed", zorder=0)
ax.axvline(1, color='grey', linewidth=0.5, linestyle="dashed", zorder=0)

ax.set_title("Exponential Grid")

plt.show()