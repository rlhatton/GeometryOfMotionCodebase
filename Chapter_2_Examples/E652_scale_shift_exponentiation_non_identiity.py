import numpy as np
from geomotion import manifold as md, diffmanifold as tb, utilityfunctions as ut, plottingfunctions as gplt, liegroup as lgp
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
# Get the flows on the generator fields, starting at the origin and the [2, -1] point
start1 = 0
end1 = 1
g_initial = G.element([2, -1])
sol_ID = G.L_generator(g_circ.value).integrate([start1, end1], G.identity_element())
sol_L = G.L_generator(g_circ.value).integrate([start1, end1], g_initial)
sol_R = G.R_generator(g_circ.value).integrate([start1, end1], g_initial)

g_initial_scale = G.element([g_initial[0], 0])

# Generate a loose set of points from zero to 1
t = np.linspace(0, 1, 5)

# Evaluate the solution trajectories at the time points
traj_ID = G.element_set(ut.GridArray(sol_ID.sol(t),1))
traj_L = G.element_set(ut.GridArray(sol_L.sol(t),1))
traj_R = G.element_set(ut.GridArray(sol_R.sol(t),1))

# Get the left- and right-scaled points along the identity-trajectory
traj_ID_R_scaled = traj_ID * g_initial_scale
traj_ID_L_scaled = g_initial_scale * traj_ID

scale_lines_L=[]
scale_lines_R=[]
for i, p in enumerate(traj_ID):
    scale_lines_L.append(lgp.LieGroupElementSet([traj_ID[i], traj_ID_R_scaled[i], traj_L[i]]))
    scale_lines_R.append(lgp.LieGroupElementSet([G.element([0, 0]), traj_ID_L_scaled[i], traj_R[i]]))

ax = plt.subplot(2, 2, 1)
c_grid_0, v_grid_0 = d_dL.grid
ax.quiver(*c_grid_0, *v_grid_0, color='black')
ax.set_xlim(0, 5.5)
ax.set_ylim(-2.5, 2.5)
ax.set_aspect('equal')
ax.plot(*traj_ID.grid, color='grey', zorder=-3, marker='.', markersize=10)
ax.plot(*traj_L.grid, color=spot_color, zorder=-3, marker='.', markersize=10)
for p in scale_lines_L:
    ax.plot(*p.grid, color='black', linewidth=0.5, linestyle="dotted", zorder=0)
ax.scatter(0, 0, edgecolor='black', facecolor='white', s=20, clip_on=False, zorder=3)
ax.axhline(0, color='grey', linewidth=0.5, linestyle="dashed", zorder=0)
ax.axvline(1, color='grey', linewidth=0.5, linestyle="dashed", zorder=0)

ax.set_title("Right-transform of left flow")

ax = plt.subplot(2, 2, 2)
c_grid_0, v_grid_0 = d_dR.grid
ax.quiver(*c_grid_0, *v_grid_0, color='black')
ax.set_xlim(0, 5.5)
ax.set_ylim(-2.5, 2.5)
ax.set_aspect('equal')
ax.plot(*traj_ID.grid, color='grey', zorder=-3, marker='.', markersize=10)
ax.plot(*traj_R.grid, color=spot_color, zorder=-3, marker='.', markersize=10)
for p in scale_lines_R:
    ax.plot(*p.grid, color='black', linewidth=0.5, linestyle="dotted", zorder=0)
ax.scatter(0, 0, edgecolor='black', facecolor='white', s=20, clip_on=False, zorder=3)
ax.axhline(0, color='grey', linewidth=0.5, linestyle="dashed", zorder=0)
ax.axvline(1, color='grey', linewidth=0.5, linestyle="dashed", zorder=0)

ax.set_title("Left-transform of right flow")

ax = plt.subplot(2, 2, 3)
c_grid_0, v_grid_0 = d_dL.grid
ax.quiver(*c_grid_0, *v_grid_0, color='black')
ax.set_xlim(0, 5.5)
ax.set_ylim(-2.5, 2.5)
ax.set_aspect('equal')
ax.plot(*traj_L.grid, color=spot_color, zorder=-3, marker='.', markersize=10)
ax.scatter(0, 0, edgecolor='black', facecolor='white', s=20, clip_on=False, zorder=3)
ax.axhline(0, color='grey', linewidth=0.5, linestyle="dashed", zorder=0)
ax.axvline(1, color='grey', linewidth=0.5, linestyle="dashed", zorder=0)
ax.set_title("Left flow from g_0")

ax = plt.subplot(2, 2, 4)
c_grid_0, v_grid_0 = d_dR.grid
ax.quiver(*c_grid_0, *v_grid_0, color='black')
ax.set_xlim(0, 5.5)
ax.set_ylim(-2.5, 2.5)
ax.set_aspect('equal')
ax.plot(*traj_R.grid, color=spot_color, zorder=-3, marker='.', markersize=10)
ax.scatter(0, 0, edgecolor='black', facecolor='white', s=20, clip_on=False, zorder=3)
ax.axhline(0, color='grey', linewidth=0.5, linestyle="dashed", zorder=0)
ax.axvline(1, color='grey', linewidth=0.5, linestyle="dashed", zorder=0)
ax.set_title("Right flow from g_0")



plt.show()
