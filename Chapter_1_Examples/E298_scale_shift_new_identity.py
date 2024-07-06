#! /usr/bin/python3
from S275_Construct_RxRplus import RxRplus as RxRplus
from geomotion import utilityfunctions as ut
from geomotion import group as gp
from geomotion import plottingfunctions as gplt
from matplotlib import pyplot as plt
import numpy as np

G = RxRplus

# Set up the initial group elements
g0 = G.identity_element()
g1 = G.element([2, 2])
g2 = G.element([2, 3])
g3 = G.element([1, 2])

initial_points = gp.GroupElementSet([g0, g1, g2, g3])

# Change the identity to g1 using a left action
primed_points = g1.inverse * initial_points
print("Moving the identity to g1 using a left action results in the points having values of \n"
      "g0=", primed_points[0], ",\n g1=", primed_points[1], ",\n g2=", primed_points[2], ",\n and g3=", primed_points[3])

dprimed_points = initial_points * g1.inverse
print(dprimed_points[0], dprimed_points[1], dprimed_points[2], dprimed_points[3])
print("\n Moving the identity to g1 using a right action results in the points having values of \n"
      "g0=", dprimed_points[0], ",\n g1=", dprimed_points[1], ",\n g2=", dprimed_points[2], ",\n and g3=", dprimed_points[3])


initial_grid = ut.meshgrid_array([0, 2, 3, 4, 5, 6], [-2, -1, 0, 1, 2, 3, 4])
group_grid = gp.GroupElementSet(RxRplus, initial_grid)
group_grid_left_transformed = g1 * group_grid
group_grid_right_transformed = group_grid * g1


##############
# Plot the calculated terms

# Use my red color for the plots
spot_color = gplt.crimson

ax_orig = plt.subplot(1, 3, 1)
ax = ax_orig
point_set = initial_points
ax.scatter(*point_set[0], edgecolor='black', facecolor='black', label='$g_{1}$')
ax.scatter(*point_set[1], edgecolor=spot_color, facecolor='white', label='$g_{1}$')
ax.scatter(*point_set[2], edgecolor='black', facecolor=spot_color, label='$g_{2}$')
ax.scatter(*point_set[3], edgecolor='black', facecolor='white', marker=r'$\odot$', label='$g_{3}$')
ax.scatter(0, 0, edgecolor='black', facecolor='white', s=20, clip_on=False, zorder=3)
ax.set_xlim(0, 6)
ax.set_ylim(-2, 4)
ax.margins(x=.5)
#ax.set_xticks([0, 1, 2, 3, 4])
#ax.set_yticks([0, 1, 2, 3, 4])
ax.set_aspect('equal')
ax.set_axisbelow(True)
xg = group_grid.grid[0]
yg = group_grid.grid[1]
ax.pcolormesh(xg, yg, np.zeros([xg.shape[0]-1, xg.shape[1]-1]), edgecolor='grey', facecolor='none', linewidth=0.25, zorder=.1)
ax.axhline(0, color='black')
ax.axvline(g0[0], color='grey', linewidth=0.5, linestyle="dashed", zorder=0)

ax_left_shifted = plt.subplot(1, 3, 2)
ax = ax_left_shifted
point_set = initial_points
ax.scatter(*point_set[0], edgecolor='black', facecolor='black', label='$g_{1}$')
ax.scatter(*point_set[1], edgecolor=spot_color, facecolor='white', label='$g_{1}$')
ax.scatter(*point_set[2], edgecolor='black', facecolor=spot_color, label='$g_{2}$')
ax.scatter(*point_set[3], edgecolor='black', facecolor='white')
ax.scatter(*point_set[3], edgecolor='black', facecolor='white', marker=r'$\odot$', label='$g_{3}$')
ax.scatter(0, g1[1], edgecolor='black', facecolor='white', s=20, clip_on=False, zorder=3)
ax.set_xlim(0, 6)
ax.set_ylim(-2, 4)
ax.margins(x=.5)
#ax.set_xticks([0, 1, 2, 3, 4])
#ax.set_yticks([0, 1, 2, 3, 4])
ax.set_aspect('equal')
ax.set_axisbelow(True)
xg = group_grid_left_transformed.grid[0]
yg = group_grid_left_transformed.grid[1]
ax.pcolormesh(xg, yg, np.zeros([xg.shape[0]-1, xg.shape[1]-1]), edgecolor='grey', facecolor='none', linewidth=0.25, zorder=.1)
ax.axhline(g1[1], color='black', zorder=0)
ax.axvline(g1[0], color='grey', linewidth=0.5, linestyle="dashed", zorder=0)

ax_right_shifted = plt.subplot(1, 3, 3)
ax = ax_right_shifted
point_set = initial_points
ax.scatter(*point_set[0], edgecolor='black', facecolor='black', label='$g_{1}$')
ax.scatter(*point_set[1], edgecolor=spot_color, facecolor='white', label='$g_{1}$')
ax.scatter(*point_set[2], edgecolor='black', facecolor=spot_color, label='$g_{2}$')
ax.scatter(*point_set[3], edgecolor='black', facecolor='white')
ax.scatter(*point_set[3], edgecolor='black', facecolor='white', marker=r'$\odot$', label='$g_{3}$')
ax.scatter(0, 0, edgecolor='black', facecolor='white', s=20, clip_on=False, zorder=3)
ax.set_xlim(0, 8)
ax.set_ylim(-2, 8)
ax.margins(x=.5)
#ax.set_xticks([0, 1, 2, 3, 4])
#ax.set_yticks([0, 1, 2, 3, 4])
ax.set_aspect('equal')
ax.set_axisbelow(True)
xg = group_grid_right_transformed.grid[0]
yg = group_grid_right_transformed.grid[1]
ax.pcolormesh(xg, yg, np.zeros([xg.shape[0]-1, xg.shape[1]-1]), edgecolor='grey', facecolor='none', linewidth=0.25, zorder=.1)
ax.plot([0, 8], [0, 8], color='black', zorder=.5)
ax.axvline(g1[0], color='grey', linewidth=0.5, linestyle="dashed", zorder=0)

plt.show()
