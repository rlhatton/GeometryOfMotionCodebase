import numpy as np
from geomotion import manifold as md, diffmanifold as tb, utilityfunctions as ut, plottingfunctions as gplt
from S550_Construct_RxRplus_rep import RxRplus
from matplotlib import pyplot as plt

# Make things print nicely
np.set_printoptions(precision=2)
spot_color = gplt.crimson

# Make the working group the scale-shift group
G = RxRplus

# Create a grid in the right-half-plane
grid_xy = ut.meshgrid_array(np.linspace(.5, 2, 4), np.linspace(-1, 1, 5))
# Turn the grid into a set of points in the group
grid_points = RxRplus.element_set(grid_xy)

# Create lists of vector fields containing the derivatives in the directions
# of the left and right group actions in each parameter
d_dL = []
d_dR = []
for i in range(G.n_dim):

    # Generate a delta from the identity along the ith coordinate
    g_delta_i_val = np.zeros_like(G.identity_list[0])
    g_delta_i_val[i] = 1
    g_delta_i = G.Lie_alg_vector(g_delta_i_val)

    # Get the group generator fields in the direction of g_delta_i,
    d_dL_i = g_delta_i * grid_points
    d_dR_i = grid_points * g_delta_i

    # Evaluate the ith directional derivatives and save them to the lists
    d_dL.append(d_dL_i)
    d_dR.append(d_dR_i)



ax = plt.subplot(1, 2, 1)
c_grid_0, v_grid_0 = d_dL[0].grid
c_grid_1, v_grid_1 = d_dL[1].grid
ax.quiver(*c_grid_0, *v_grid_0, scale=20, color=spot_color)
ax.quiver(*c_grid_1, *v_grid_1, scale=20, color='black')
ax.set_aspect('equal')
ax.set_xlim(0, 2.25)
ax.set_ylim(-1.1, 1.25)
# ax.scatter(1, 0, edgecolor='black', facecolor='black', zorder=-2)
ax.scatter(0, 0, edgecolor='black', facecolor='white', s=20, clip_on=False, zorder=3)
ax.axhline(0, color='grey', linewidth=0.5, linestyle="dashed", zorder=0)
ax.axvline(1, color='grey', linewidth=0.5, linestyle="dashed", zorder=0)
ax.plot([0, c_grid_0[0][0][0]], [0, c_grid_0[1][0][0]], color='grey', linewidth=0.5, linestyle="dotted", zorder=0)
ax.plot([0, c_grid_0[0][1][2]], [0, c_grid_0[1][1][2]], color='grey', linewidth=0.5, linestyle="dotted", zorder=0)
ax.plot([0, c_grid_0[0][-1][-1]], [0, c_grid_0[1][-1][-1]], color='grey', linewidth=0.5, linestyle="dotted", zorder=0)
ax.plot([0, c_grid_0[0][-1][1]], [0, c_grid_0[1][-1][1]], color='grey', linewidth=0.5, linestyle="dotted", zorder=0)

ax.set_title("Left basis vectors")

ax = plt.subplot(1, 2, 2)
c_grid_0, v_grid_0 = d_dR[0].grid
c_grid_1, v_grid_1 = d_dR[1].grid
ax.quiver(*c_grid_0, *v_grid_0, scale=20, color=spot_color)
ax.quiver(*c_grid_1, *v_grid_1, scale=20, color='black')
ax.set_aspect('equal')
ax.set_xlim(0, 2.25)
ax.set_ylim(-1.1, 1.25)
# ax.scatter(1, 0, edgecolor='black', facecolor='black', zorder=-2)
ax.scatter(0, 0, edgecolor='black', facecolor='white', s=20, clip_on=False, zorder=3)
ax.axhline(0, color='grey', linewidth=0.5, linestyle="dashed", zorder=0)
ax.axvline(1, color='grey', linewidth=0.5, linestyle="dashed", zorder=0)

ax.set_title("Right basis vectors")

plt.show()
