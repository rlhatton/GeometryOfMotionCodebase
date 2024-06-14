#! /usr/bin/python3
from S280_Construct_RxRplus_rep import RxRplus
from geomotion import plottingfunctions as gplt
from matplotlib import pyplot as plt
import numpy as np

np.set_printoptions(precision=2)  # Make things print nicely

# Set the working group to be the semi-direct-product scale-shift group
G = RxRplus

g1 = RxRplus.element([3, -1])
g2 = RxRplus.element([.5, 1.5])

g3 = g1 * g2

print("Left scale-shift action of ", g1, " on ", g2, " is ", g3)

g4 = g2 * g1

print("Right scale-shift action of ", g1, " on ", g2, " is ", g4)

print("Inverse of g1 is ", g1.inverse)

g_delta_right = g1.inverse * g3
print("Left inverse scale-shift action of ", g1, " on ", g2, " is ", g_delta_right)



##########
# Plotting code

# Calculate some intermediate points for drawing arrows
g1scale = RxRplus.element([3, 0])
g1scale_g2 = g1scale * g2

g2shift = RxRplus.element([1, 1.5])
g1_g2shift = g1 * g2shift

g2scale = RxRplus.element([.5, 0])
g2scale_g1 = g2scale * g1

g1shift = RxRplus.element([1, -1])
g2_g1shift = g2 * g1shift


spot_color=gplt.crimson

ax_orig = plt.subplot(3, 1, 1)
ax = ax_orig
ax.scatter(g1[0], g1[1], edgecolor=spot_color, facecolor='white', label='$g_{1}$')
ax.scatter(g2[0], g2[1], edgecolor='black', facecolor=spot_color, label='$g_{2}$')
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

ax_g1g2_left = plt.subplot(3, 2, 3)
ax = ax_g1g2_left
ax.scatter(g1[0], g1[1], edgecolor=spot_color, facecolor='white')
ax.scatter(g2[0], g2[1], edgecolor='black', facecolor=spot_color)
ax.scatter(g3[0], g3[1], edgecolor='black', facecolor='white')
ax.scatter(g3[0], g3[1], edgecolor='black', facecolor='black', s=3, zorder=3)
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
ax.arrow(g2[0], g2[1], g1scale_g2[0]-g2[0], g1scale_g2[1]-g2[1], color='black', zorder=.9, length_includes_head=True, width=.01, head_width=0.2, overhang=.5)
ax.arrow(g1scale_g2[0], g1scale_g2[1], .8*(g1scale_g2[0]-g3[0]), .8*(g3[1]-g1scale_g2[1]), color='black', zorder=.9, length_includes_head=True,  width=.01, head_width=0.2, overhang=.5)
ax.plot([1, g1scale[0], g1[0]], [0-.2, g1scale[1]-.2, g1[1]], color='black', linewidth=.5, linestyle='dashed', zorder=-1)

ax_g1g2_right = plt.subplot(3, 2, 4)
ax = ax_g1g2_right
ax.scatter(g1[0], g1[1], edgecolor=spot_color, facecolor='white')
ax.scatter(g2[0], g2[1], edgecolor='black', facecolor=spot_color)
ax.scatter(g3[0], g3[1], edgecolor='black', facecolor='white')
ax.scatter(g3[0], g3[1], edgecolor='black', facecolor='black', s=3, zorder=3)
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
ax.arrow(g1[0], g1[1], g1_g2shift[0]-g1[0], g1_g2shift[1]-g1[1], color='black', zorder=.9, length_includes_head=True, width=.01, head_width=0.2, overhang=.5)
ax.arrow(g1_g2shift[0], g1_g2shift[1], .8*(g3[0]-g1_g2shift[0]), g3[1]-g1_g2shift[1], color='black', zorder=.9, length_includes_head=True,  width=.01, head_width=0.2, overhang=.5)
ax.plot([1, g2shift[0], g2[0]], [0, g2shift[1], g2[1]], color='black', linewidth=.5, linestyle='dashed', zorder=-1)

ax_g2g1_left = plt.subplot(3, 2, 5)
ax = ax_g2g1_left
ax.scatter(g1[0], g1[1], edgecolor=spot_color, facecolor='white')
ax.scatter(g2[0], g2[1], edgecolor='black', facecolor=spot_color)
ax.scatter(g4[0], g4[1], edgecolor=spot_color, facecolor='white')
ax.scatter(g4[0], g4[1], edgecolor=spot_color, facecolor='black', s=3, zorder=3)
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
ax.plot([0, g2scale_g1[0]], [0, g2scale_g1[1]], color='black', linewidth=.5, linestyle='dashed', zorder=-1)
ax.arrow(g1[0], g1[1], g2scale_g1[0]-g1[0], g2scale_g1[1]-g1[1], color='black', zorder=.9, length_includes_head=True, width=.01, head_width=0.2, overhang=.5)
ax.arrow(g2scale_g1[0], g2scale_g1[1], .8*(g4[0]-g2scale_g1[0]), .8*(g4[1]-g2scale_g1[1]), color='black', zorder=.9, length_includes_head=True,  width=.01, head_width=0.2, overhang=.5)
ax.plot([1, g2scale[0], g2[0]], [0+.2, g2scale[1]+.2, g2[1]], color='black', linewidth=.5, linestyle='dashed', zorder=-1)

ax_g2g1_right = plt.subplot(3, 2, 6)
ax = ax_g2g1_right
ax.scatter(g1[0], g1[1], edgecolor=spot_color, facecolor='white')
ax.scatter(g2[0], g2[1], edgecolor='black', facecolor=spot_color)
ax.scatter(g4[0], g4[1], edgecolor=spot_color, facecolor='white')
ax.scatter(g4[0], g4[1], edgecolor=spot_color, facecolor='black', s=3, zorder=3)
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
ax.arrow(g2[0], g2[1], g2_g1shift[0]-g2[0], g2_g1shift[1]-g2[1], color='black', zorder=.9, length_includes_head=True, width=.01, head_width=0.2, overhang=.5)
ax.arrow(g2_g1shift[0], g2_g1shift[1], .8*(g4[0]-g2_g1shift[0]), g4[1]-g2_g1shift[1], color='black', zorder=.9, length_includes_head=True,  width=.01, head_width=0.2, overhang=.5)
ax.plot([1, g1shift[0], g1[0]], [0, g1shift[1], g1[1]], color='black', linewidth=.5, linestyle='dashed', zorder=-1)


plt.show()