#! /usr/bin/python3
from S280_Construct_RxRplus_rep import RxRplus as RxRplus
from geomotion import group as gp
from geomotion import plottingfunctions as gplt
from matplotlib import pyplot as plt
import numpy as np

G = RxRplus

# Set up the initial group elements
g1 = G.element([3, -1])
g2 = G.element([.5, 1.5])

Comm_g1_g2 = gp.commutator(g1, g2)
print("The commutator of g1=", g1, " and g2=", g2, " is ", Comm_g1_g2, "\n")

g1g2 = g1 * g2
g2g1 = g2 * g1
print("This commutator can be interpreted as the left difference between g1*g2 =", g1g2, " and g2*g1=", g2g1,
      ", \ni.e., the group element whose left action takes g2*g1 to g1*g2.\n \n"
      "Alternatively, the commutator can be considered as the point reached through the sequence g1 g2 g1inv g2inv.")




######
# Plotting

spot_color = gplt.crimson

g1g2g1inv = g1g2 * g1.inverse

g1_shift = G.element([1, g1[1]])
e_g1_shift = G.identity_element() * g1_shift
g2_g1_shift = g2 * g1_shift

g2_shift = G.element([1, g2[1]])
g1_g2_shift = g1 * g2_shift

g1inv_shift = G.element([1, g1.inverse[1]])
g1g2_g1inv_shift = g1g2 * g1inv_shift

g2inv_shift = G.element([1, g2.inverse[1]])
g1g2g1inv_g2inv_shift = g1g2g1inv * g2inv_shift


ax_CommDiff = plt.subplot(1, 2, 1)
ax = ax_CommDiff
ax.scatter(g1[0], g1[1], edgecolor=spot_color, facecolor='white')
ax.scatter(g2[0], g2[1], edgecolor='black', facecolor=spot_color)
ax.scatter(g1g2[0], g1g2[1], edgecolor='black', facecolor='white', marker=r'$\odot$')
ax.scatter(g2g1[0], g2g1[1], edgecolor=spot_color, facecolor='white', marker=r'$\odot$')
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
# ax.plot([0, g2[0]], [0, g2[1]], color='black', linewidth=.5, linestyle='dashed', zorder=-1)
ax.arrow(g1[0], g1[1], g1_g2_shift[0]-g1[0], g1_g2_shift[1]-g1[1], color='black', zorder=.9, length_includes_head=True, width=.01, head_width=0.2, overhang=.5)
ax.arrow(g1_g2_shift[0], g1_g2_shift[1], .9*(g1g2[0]-g1_g2_shift[0]), .9*(g1g2[1]-g1_g2_shift[1]), color='black', zorder=.9, length_includes_head=True,  width=.01, head_width=0.2, overhang=.5)
ax.arrow(g2[0], g2[1], g2_g1_shift[0]-g2[0], g2_g1_shift[1]-g2[1], color='black', zorder=.9, length_includes_head=True, width=.01, head_width=0.2, overhang=.5)
ax.arrow(g2_g1_shift[0], g2_g1_shift[1], .9*(g2g1[0]-g2_g1_shift[0]), g2g1[1]-g2_g1_shift[1], color='black', zorder=.9, length_includes_head=True, width=.01, head_width=0.2, overhang=.5)
ax.arrow(g2g1[0], g2g1[1]+.1, .9*(g1g2[0]-g2g1[0]), .9*(g1g2[1]-g2g1[1]), color=spot_color, zorder=.75, length_includes_head=True,  width=.01, head_width=0.2, overhang=.5)
# ax.plot([1, g1scale[0], g1[0]], [0-.2, g1scale[1]-.2, g1[1]], color='black', linewidth=.5, linestyle='dashed', zorder=-1)

ax_CommPath = plt.subplot(1, 2, 2)
ax = ax_CommPath
ax.scatter(g1[0], g1[1], edgecolor=spot_color, facecolor='white')
# ax.scatter(g2[0], g2[1], edgecolor='black', facecolor=spot_color)
ax.scatter(g1g2[0], g1g2[1], edgecolor='black', facecolor='white', marker=r'$\odot$')
ax.scatter(g1g2g1inv[0], g1g2g1inv[1], edgecolor=spot_color, facecolor='black')
ax.scatter(Comm_g1_g2[0], Comm_g1_g2[1], edgecolor=spot_color, facecolor='white', marker=r'$\otimes$')
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
# ax.plot([0, g2[0]], [0, g2[1]], color='black', linewidth=.5, linestyle='dashed', zorder=-1)
ax.arrow(g1[0], g1[1], g1_g2_shift[0]-g1[0], g1_g2_shift[1]-g1[1], color='black', zorder=.9, length_includes_head=True, width=.01, head_width=0.2, overhang=.5)
ax.arrow(g1_g2_shift[0], g1_g2_shift[1], .9*(g1g2[0]-g1_g2_shift[0]), .9*(g1g2[1]-g1_g2_shift[1]), color='black', zorder=.9, length_includes_head=True,  width=.01, head_width=0.2, overhang=.5)

ax.arrow(g1g2[0], g1g2[1]+.1, g1g2_g1inv_shift[0]-g1g2[0], .8*(g1g2_g1inv_shift[1]-g1g2[1]), color='black', zorder=.9, length_includes_head=True, width=.01, head_width=0.2, overhang=.5)
ax.arrow(g1g2_g1inv_shift[0], g1g2_g1inv_shift[1], .9*(g1g2g1inv[0]-g1g2_g1inv_shift[0]), g1g2g1inv[1]-g1g2_g1inv_shift[1], color='black', zorder=.9, length_includes_head=True, width=.01, head_width=0.2, overhang=.5)

ax.arrow(g1g2g1inv[0], g1g2g1inv[1]-.1, g1g2g1inv_g2inv_shift[0]-g1g2g1inv[0], .9*(g1g2g1inv_g2inv_shift[1]-g1g2g1inv[1]), color='black', zorder=.9, length_includes_head=True, width=.01, head_width=0.2, overhang=.5)
ax.arrow(g1g2g1inv_g2inv_shift[0], g1g2g1inv_g2inv_shift[1], .9*(Comm_g1_g2[0]-g1g2g1inv_g2inv_shift[0]), Comm_g1_g2[1]-g1g2g1inv_g2inv_shift[1], color='black', zorder=.9, length_includes_head=True, width=.01, head_width=0.2, overhang=.5)

ax.arrow(1,0, .95*(g1g2[0]-g2g1[0]), .95*(g1g2[1]-g2g1[1]), color=spot_color, zorder=.75, length_includes_head=True,  width=.01, head_width=0.2, overhang=.5)
# ax.plot([1, g1scale[0], g1[0]], [0-.2, g1scale[1]-.2, g1[1]], color='black', linewidth=.5, linestyle='dashed', zorder=-1)

plt.show()

#
# ax_g2 = plt.subplot(1, 3, 2)
# ax = ax_g2
# ax.scatter(g2[0], g2[1], edgecolor=spot_color, facecolor='white', label='$g_{1}$')
# ax.scatter(g2_h1[0], g2_h1[1], edgecolor='black', facecolor=spot_color, label='$g_{2}$')
# ax.scatter(1,0, edgecolor='black', facecolor='black')
# ax.scatter(0, 0, edgecolor='black', facecolor='white', s=20, clip_on=False, zorder=3)
# ax.set_xlim(0, 3)
# ax.set_ylim(-1, 5)
# ax.margins(x=.5)
# #ax.set_xticks([0, 1, 2, 3, 4])
# #ax.set_yticks([0, 1, 2, 3, 4])
# ax.set_aspect('equal')
# ax.grid(True)
# ax.set_axisbelow(True)
# ax.axhline(0, color='black')
# ax.arrow(g2[0], g2[1], .8*(g2_h1[0] - g2[0]), .8*(g2_h1[1] - g2[1]), color='black', zorder=.9, length_includes_head=True, width=.01, head_width=0.2, overhang=.5)
# ax.arrow(g2[0], g2[1], (h3scale_g2[0] - g2[0]), (h3scale_g2[1] - g2[1]), color=spot_color, zorder=.9, length_includes_head=True, width=.01, head_width=0.2, overhang=.5)
# ax.arrow(h3scale_g2[0], h3scale_g2[1], .9 * (g2_h1[0] - h3scale_g2[0]), .9 * (g2_h1[1] - h3scale_g2[1]), color=spot_color, zorder=.9, length_includes_head=True, width=.01, head_width=0.2, overhang=.5)
# ax.plot([0, h3scale_g2[0]], [0, h3scale_g2[1]], color='black', linewidth=.5, linestyle='dashed', zorder=-1)
#
# ax_h4 = plt.subplot(1, 3, 3)
# ax = ax_h4
# ax.scatter(g1[0], g1[1], edgecolor=spot_color, facecolor='white', label='$g_{1}$')
# ax.scatter(g1_h2[0], g1_h2[1], edgecolor='black', facecolor=spot_color, label='$g_{2}$')
# ax.scatter(1,0, edgecolor='black', facecolor='black')
# ax.scatter(0, 0, edgecolor='black', facecolor='white', s=20, clip_on=False, zorder=3)
# ax.set_xlim(0, 3)
# ax.set_ylim(-1, 5)
# ax.margins(x=.5)
# #ax.set_xticks([0, 1, 2, 3, 4])
# #ax.set_yticks([0, 1, 2, 3, 4])
# ax.set_aspect('equal')
# ax.grid(True)
# ax.set_axisbelow(True)
# ax.axhline(0, color='black')
# ax.arrow(g1[0], g1[1], (g1_h2shift[0] - g1[0]), (g1_h2shift[1] - g1[1]), color='black', zorder=.9, length_includes_head=True, width=.01, head_width=0.2, overhang=.5)
# ax.arrow(g1_h2shift[0], g1_h2shift[1], .85*(g1_h2[0] - g1_h2shift[0]), .85*(g1_h2[1] - g1_h2shift[1]), color='black', zorder=.9, length_includes_head=True, width=.01, head_width=0.2, overhang=.5)
# ax.arrow(g1[0], g1[1], (h4scale_g1[0] - g1[0]), (h4scale_g1[1] - g1[1]), color=spot_color, zorder=.9, length_includes_head=True, width=.01, head_width=0.2, overhang=.5)
# ax.arrow(h4scale_g1[0], h4scale_g1[1], .95 * (g1_h2[0] - h4scale_g1[0]), .95 * (g1_h2[1] - h4scale_g1[1]), color=spot_color, zorder=.9, length_includes_head=True, width=.01, head_width=0.2, overhang=.5)
# ax.plot([0, h2scale_g1[0]], [0, h2scale_g1[1]], color='black', linewidth=.5, linestyle='dashed', zorder=-1)
#
#
# plt.show()