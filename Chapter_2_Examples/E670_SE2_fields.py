import numpy as np
from geomotion import utilityfunctions as ut, plottingfunctions as gplt
from S670_Construct_SE2 import SE2
from matplotlib import pyplot as plt

# Make things print nicely
np.set_printoptions(precision=2)
spot_color = gplt.crimson

# Make the working group the scale-shift group
G = SE2

# Create a grid of points
grid_xy = ut.meshgrid_array(np.linspace(-1, 1, 2), np.linspace(-1, 1, 2), np.linspace(0, np.pi/4, 2))
# Turn the grid into a set of points in the group
grid_points = G.element_set(grid_xy)

# Create lists of vector fields containing the derivatives in the directions
# of the left and right group actions in each parameter
g_circ_R_field = []
g_circ_L_field = []
for i in range(G.n_dim):
    # Generate a unit velocity along one coordinate direction at the identity
    g_circ_i_val = np.zeros_like(G.identity_list[0])
    g_circ_i_val[i] = 1
    g_circ_i = G.vector(G.identity_element(), g_circ_i_val)

    # Create vector fields from the lifted actions acting on g_circ_i
    g_circ_R_field_i = grid_points * g_circ_i
    g_circ_L_field_i = g_circ_i * grid_points

    # Evaluate the ith directional derivatives and save them to the lists
    g_circ_R_field.append(g_circ_R_field_i)
    g_circ_L_field.append(g_circ_L_field_i)

ax = plt.subplot(1, 2, 1, projection='3d')
c_grid_0, v_grid_0 = g_circ_L_field[0].grid
c_grid_1, v_grid_1 = g_circ_L_field[1].grid
c_grid_2, v_grid_2 = g_circ_L_field[2].grid
ax.quiver(*c_grid_0, *v_grid_0, color=spot_color)
ax.quiver(*c_grid_1, *v_grid_1, color='black')
ax.quiver(*c_grid_2, *v_grid_2, color='grey')
ax.set_aspect('equal')
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
# ax.scatter(1, 0, edgecolor='black', facecolor='black', zorder=-2)
ax.scatter(0, 0, edgecolor='black', facecolor='white', s=20, clip_on=False, zorder=3)
#ax.axhline(0, color='grey', linewidth=0.5, linestyle="dashed", zorder=0)
#ax.axvline(1, color='grey', linewidth=0.5, linestyle="dashed", zorder=0)

ax.set_title("Spatial (left) velocity field")


ax = plt.subplot(1, 2, 2, projection='3d')
c_grid_0, v_grid_0 = g_circ_R_field[0].grid
c_grid_1, v_grid_1 = g_circ_R_field[1].grid
c_grid_2, v_grid_2 = g_circ_R_field[2].grid
ax.quiver(*c_grid_0, *v_grid_0, color=spot_color)
ax.quiver(*c_grid_1, *v_grid_1, color='black')
ax.quiver(*c_grid_2, *v_grid_2, color='grey')
ax.set_aspect('equal')
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
# ax.scatter(1, 0, edgecolor='black', facecolor='black', zorder=-2)
ax.scatter(0, 0, edgecolor='black', facecolor='white', s=20, clip_on=False, zorder=3)
#ax.axhline(0, color='grey', linewidth=0.5, linestyle="dashed", zorder=0)
#ax.axvline(1, color='grey', linewidth=0.5, linestyle="dashed", zorder=0)

ax.set_title("Body (right) velocity field")



plt.show()
