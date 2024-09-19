import numpy as np
from geomotion import manifold as md, diffmanifold as tb, utilityfunctions as ut, plottingfunctions as gplt
from S500_Construct_RxRplus import RxRplus
from matplotlib import pyplot as plt

# Make things print nicely
np.set_printoptions(precision=2)
spot_color = gplt.crimson

# Make the working group the scale-shift group
G = RxRplus

# Create a grid in the right-half-plane
grid_xy = ut.meshgrid_array(np.linspace(.5, 5, 10), np.linspace(-2, 2, 9))
# Turn the grid into a set of points in the group
grid_points = RxRplus.element_set(grid_xy)

# Create left and right generator fields for the [1, 1] vector
g_circ = G.Lie_alg_vector([1, 1])
d_dL = g_circ * grid_points
d_dR = grid_points * g_circ

###
# Exponentiate g_circ as both a left and right groupwise velocity, to demonstrate that they are the same
g_circ_exp_L = g_circ.exp_L
g_circ_exp_R =g_circ.exp_R


print("The left exponential of g_circ=", g_circ, " is ", g_circ_exp_L, ", which is the"
                                                                       " same as its right exponential, ", g_circ_exp_R)

###
# For display, get the flows on the generator fields, starting at the origin
start1 = 0
end1 = 1
g_initial = G.identity_element()
solL = G.L_generator(g_circ.value).integrate([start1, end1], g_initial)
solR = G.R_generator(g_circ.value).integrate([start1, end1], g_initial)

# Generate a loose set of points from zero to 1
t = np.linspace(0, 1, 5)

# Evaluate the solution trajectories at the time points
trajL = solL.sol(t)
trajR = solR.sol(t)

ax = plt.subplot(1, 2, 1)
c_grid_0, v_grid_0 = d_dL.grid
ax.quiver(*c_grid_0, *v_grid_0, color='black')
ax.set_xlim(0, 5.5)
ax.set_xticks([1, 3, 5])
ax.set_ylim(-2.5, 2.5)
ax.set_aspect('equal')
ax.scatter(*g_circ_exp_L, edgecolor=spot_color, facecolor=spot_color, zorder=-2)
ax.plot(*trajL, color='grey', zorder=-3, marker='.', markersize=10)
ax.scatter(0, 0, edgecolor='black', facecolor='white', s=20, clip_on=False, zorder=3)
ax.axhline(0, color='grey', linewidth=0.5, linestyle="dashed", zorder=0)
ax.axvline(1, color='grey', linewidth=0.5, linestyle="dashed", zorder=0)

ax.set_title("Left generator field")

ax = plt.subplot(1, 2, 2)
c_grid_0, v_grid_0 = d_dR.grid
ax.quiver(*c_grid_0, *v_grid_0, color='black')
ax.set_xlim(0, 5.5)
ax.set_xticks([1, 3, 5])
ax.set_ylim(-2.5, 2.5)
ax.set_aspect('equal')
ax.scatter(*g_circ_exp_R, edgecolor=spot_color, facecolor=spot_color, zorder=-2)
ax.plot(*trajR, color='grey', zorder=-3, marker='.', markersize=10)
ax.scatter(0, 0, edgecolor='black', facecolor='white', s=20, clip_on=False, zorder=3)
ax.axhline(0, color='grey', linewidth=0.5, linestyle="dashed", zorder=0)
ax.axvline(1, color='grey', linewidth=0.5, linestyle="dashed", zorder=0)

ax.set_title("Right generator field")

plt.show()
