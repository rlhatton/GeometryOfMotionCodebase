import numpy as np
from geomotion import utilityfunctions as ut, plottingfunctions as gplt
from S670_Construct_SE2 import SE2, RigidBody, cornered_triangle
from matplotlib import pyplot as plt

# Make things print nicely
np.set_printoptions(precision=2)
spot_color = gplt.crimson

# Make the working group the SE(2) group
G = SE2

# Create a set of points
g_points = G.element_set(ut.GridArray([[0, 0, 0], [1, 1, np.pi/4], [1, -1, -np.pi/2], [-1, -1, np.pi/4], [-1, 1, np.pi/2]], 1))

# Create two Lie algebra vectors
g_circ_L = G.Lie_alg_vector([1, .5, 1])
g_circ_R = G.Lie_alg_vector([1, 0, 0])

# Create lists of vector fields containing the derivatives in the directions
# of the left and right group actions in each parameter
g_circ_L_field = g_circ_L * g_points
g_circ_R_field = g_points * g_circ_R

# Create rigid bodies at each of the g_points
triangles = []
for g in g_points:
    triangles.append(RigidBody(cornered_triangle(.25, spot_color), g))

ax = plt.subplot(1, 2, 1)
c_grid_0, v_grid_0 = g_circ_L_field.grid
ax.quiver(*c_grid_0[:2], *v_grid_0[:2], color='black', scale=10, zorder=2)
ax.set_aspect('equal')
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
# ax.scatter(1, 0, edgecolor='black', facecolor='black', zorder=-2)
#ax.scatter(0, 0, edgecolor='black', facecolor='white', s=20, clip_on=False, zorder=3)
ax.scatter(-.5, 1, edgecolor='black', facecolor='white', s=20, clip_on=False, zorder=4)
ax.axhline(0, color='grey', linewidth=0.5, linestyle="dashed", zorder=0)
ax.axvline(0, color='grey', linewidth=0.5, linestyle="dashed", zorder=0)

for g in g_points:
    ax.plot([-.5, g.value[0]], [1, g.value[1]], color='grey', linewidth=0.5, linestyle="dotted", zorder=3)

for t in triangles:
    t.draw(ax)

ax.set_title("Spatial (left) velocity field")


ax = plt.subplot(1, 2, 2)
c_grid_0, v_grid_0 = g_circ_R_field.grid
ax.quiver(*c_grid_0[:2], *v_grid_0[:2], color='black', scale=5, zorder=2)
ax.set_aspect('equal')
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
# ax.scatter(1, 0, edgecolor='black', facecolor='black', zorder=-2)
#ax.scatter(0, 0, edgecolor='black', facecolor='white', s=20, clip_on=False, zorder=3)
ax.axhline(0, color='grey', linewidth=0.5, linestyle="dashed", zorder=0)
ax.axvline(0, color='grey', linewidth=0.5, linestyle="dashed", zorder=0)

for t in triangles:
    t.draw(ax)

ax.set_title("Body (right) velocity field")



plt.show()
