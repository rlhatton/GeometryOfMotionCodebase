#! /usr/bin/python3
from S280_Construct_RxRplus_rep import RxRplus as RxRplus_rep
from geomotion import plottingfunctions as gplt
from matplotlib import pyplot as plt
import numpy as np

np.set_printoptions(precision=2)  # Make things print nicely

# Set the working group to be the semi-direct-product scale-shift group
G = RxRplus_rep

g1 = G.element([3, -1])
g2 = G.element([.5, 1.5])
g3 = G.element([1.5, 3.5])

g_delta_left = g3 * g2.inverse
print("Left difference between g3=", g3, " and g2=", g2, " is ", g_delta_left, ", which in this case is equal to g1")

g_delta_right = g1.inverse * g3
print("Left difference between g3=", g3, " and g1=", g1, " is ", g_delta_right, ", which in this case is equal to g2")

##########
# Plotting code

# Calculate some intermediate points for drawing arrows

g1scale = G.element([3, 0])
g1scale_g2 = g1scale * g2

g2shift = G.element([1, 1.5])
g1_g2shift = g1 * g2shift

g2scale = G.element([.5, 0])
g2scale_g1 = g2scale * g1

g1shift = G.element([1, -1])
g2_g1shift = g2 * g1shift

g2inv_shift = G.element([1, g2.inverse[1]])
g3_g2inv_shift = g3 * g2inv_shift

g1inv_scale = G.element([g1.inverse[0], 0])
g1inv_scale_g3 = g1inv_scale * g3

spot_color = gplt.crimson

ax_orig = plt.subplot(2, 1, 1)
ax = ax_orig
ax.scatter(g1[0], g1[1], edgecolor=spot_color, facecolor='white', label='$g_{1}$')
ax.scatter(g2[0], g2[1], edgecolor='black', facecolor=spot_color, label='$g_{2}$')
ax.scatter(g3[0], g3[1], edgecolor='black', facecolor='white', label='$g_{3}$', marker=r'$\odot$')
ax.scatter(1,0, edgecolor='black', facecolor='black')
ax.scatter(0, 0, edgecolor='black', facecolor='white', s=20, clip_on=False, zorder=3)
ax.set_xlim(0, 4)
ax.set_ylim(-2, 5)
ax.margins(x=.5)
#ax.set_xticks([0, 1, 2, 3, 4])
#ax.set_yticks([0, 1, 2, 3, 4])
ax.set_aspect('equal')
ax.grid(True)
ax.set_axisbelow(True)
ax.axhline(0, color='black')
ax.legend(loc='upper right', bbox_to_anchor=(2, 1))

ax_g_delta_left = plt.subplot(2, 2, 3)
ax = ax_g_delta_left
ax.scatter(g1[0], g1[1], edgecolor=spot_color, facecolor='white')
ax.scatter(g2[0], g2[1], edgecolor='black', facecolor=spot_color)
ax.scatter(g3[0], g3[1], edgecolor='black', facecolor='white', marker=r'$\odot$')
ax.scatter(1,0, edgecolor='black', facecolor='black')
ax.scatter(0, 0, edgecolor='black', facecolor='white', s=20, clip_on=False, zorder=3)
ax.set_xlim(0, 4)
ax.set_ylim(-2, 5)
ax.margins(x=.5)
#ax.set_xticks([0, 1, 2, 3, 4])
#ax.set_yticks([0, 1, 2, 3, 4])
ax.set_aspect('equal')
ax.grid(True)
ax.set_axisbelow(True)
ax.axhline(0, color='black')
ax.plot([0, g2[0]], [0, g2[1]], color='black', linewidth=.5, linestyle='dashed', zorder=-1)
ax.arrow(g2[0], g2[1], g1scale_g2[0]-g2[0], g1scale_g2[1]-g2[1], color=spot_color, zorder=.9, length_includes_head=True, width=.01, head_width=0.2, overhang=.5)
ax.arrow(g1scale_g2[0], g1scale_g2[1], .8*(g3[0]-g1scale_g2[0]), .8*(g3[1]-g1scale_g2[1]), color=spot_color, zorder=.9, length_includes_head=True,  width=.01, head_width=0.2, overhang=.5)
ax.arrow(g2[0], g2[1], g1scale_g2[0]-g2[0], g1scale_g2[1]-g2[1], color=spot_color, zorder=.9, length_includes_head=True, width=.01, head_width=0.2, overhang=.5)
ax.arrow(1, 0-.2, (g1[0]-1), 0, color=spot_color, zorder=.9, length_includes_head=True,  width=.01, head_width=0.2, overhang=.5)
ax.arrow(g1[0], 0-.2, 0, .7*(g1[1]), color=spot_color, zorder=.9, length_includes_head=True,  width=.01, head_width=0.2, overhang=.5)
ax.arrow(g3[0], g3[1], (g3_g2inv_shift[0] - g3[0]), (g3_g2inv_shift[1] - g3[1]), color='black', zorder=.9, length_includes_head=True, width=.01, head_width=0.2, overhang=.5)
ax.arrow(g3_g2inv_shift[0], g3_g2inv_shift[1], .8*(g1[0] - g3_g2inv_shift[0]), (g1[1] - g3_g2inv_shift[1]), color='black', zorder=.9, length_includes_head=True, width=.01, head_width=0.2, overhang=.5)

ax_g_delta_left = plt.subplot(2, 2, 4)
ax = ax_g_delta_left
ax.scatter(g1[0], g1[1], edgecolor=spot_color, facecolor='white')
ax.scatter(g2[0], g2[1], edgecolor='black', facecolor=spot_color)
ax.scatter(g3[0], g3[1], edgecolor='black', facecolor='white', marker=r'$\odot$')
ax.scatter(1,0, edgecolor='black', facecolor='black')
ax.scatter(0, 0, edgecolor='black', facecolor='white', s=20, clip_on=False, zorder=3)
ax.set_xlim(0, 4)
ax.set_ylim(-2, 5)
ax.margins(x=.5)
#ax.set_xticks([0, 1, 2, 3, 4])
#ax.set_yticks([0, 1, 2, 3, 4])
ax.set_aspect('equal')
ax.grid(True)
ax.set_axisbelow(True)
ax.axhline(0, color='black')


ax.plot([0, g2[0]], [0, g2[1]], color='black', linewidth=.5, linestyle='dashed', zorder=-1)
ax.plot([0, g1inv_scale_g3[0]], [0, g1inv_scale_g3[1]], color='black', linewidth=.5, linestyle='dashed', zorder=-1)

# g1 to g3
ax.arrow(g1[0], g1[1], g1_g2shift[0]-g1[0], g1_g2shift[1]-g1[1], color=spot_color, zorder=.9, length_includes_head=True, width=.01, head_width=0.2, overhang=.5)
ax.arrow(g1_g2shift[0], g1_g2shift[1], .8*(g3[0]-g1_g2shift[0]), .8*(g3[1]-g1_g2shift[1]), color=spot_color, zorder=.9, length_includes_head=True,  width=.01, head_width=0.2, overhang=.5)

# e to g2
ax.arrow(1, 0, 0, g2[1], color=spot_color, zorder=.9, length_includes_head=True,  width=.01, head_width=0.2, overhang=.5)
ax.arrow(1, g2[1], .7*(g2[0]-1), 0, color=spot_color, zorder=.9, length_includes_head=True,  width=.01, head_width=0.2, overhang=.5)

# g3 to g2
ax.arrow(g3[0], g3[1], (g1inv_scale_g3[0] - g3[0]), (g1inv_scale_g3[1] - g3[1]), color='black', zorder=.9, length_includes_head=True, width=.01, head_width=0.2, overhang=.5)
ax.arrow(g1inv_scale_g3[0], g1inv_scale_g3[1], .8*(g2[0] - g1inv_scale_g3[0]), (g2[1] - g1inv_scale_g3[1]), color='black', zorder=.9, length_includes_head=True, width=.01, head_width=0.2, overhang=.5)


plt.show()