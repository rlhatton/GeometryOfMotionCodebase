#! /usr/bin/python3
from S280_Construct_RxRplus_rep import RxRplus as RxRplus
from geomotion import plottingfunctions as gplt
from matplotlib import pyplot as plt
import numpy as np

G = RxRplus

# Set up the initial group elements
g1 = G.element([2, 2])
h1 = G.element([0.5, 0])

# Construct the adjoint of h1 at g1
h2 = g1.AD(h1)

print("The adjoint of h1=", h1, " at g1=", g1, " is h2=AD_g1(h1)=", h2, ",\n which satisfies g1*h1=", g1*h1,
      "being equal to h2*g1=", h2*g1, "\n")

# Take the adjoint at a different point
# Set up the initial group elements
g2 = G.element([2, 3])

# Construct the adjoint of h1 at g2
h3 = g2.AD(h1)

print("The adjoint of h1=", h1, " at g2=", g2, " is h3=AD_g2(h1)=", h3, ",\n which satisfies g2*h1=", g2*h1,
      "being equal to h3*g2=", h3*g2, "\n")


# Demonstrate that the adjoint of the adjoint is not the original group element

# Calculate the adjoint of h2 at g1
h4 = g1.AD(h2)

# Calculate the adjoint-inverse of h2 at g1
h5 = g1.AD_inv(h2)

print("The adjoint of h2=", h2, " at g1=", g1, " is h4=AD_g1(h2)=", h4, ",\n which satisfies g1*h2=", g1*h2,
      "being equal to h4*g1=", h4*g1, ".\n\n", "Note h1 and h4 are different because taking adjoint twice is not "
      "equivalent\n to taking and undoing the adjoint operation. To undo the adjoint operation,\n we can use the"
      "adjoint-inverse operation h5=Adinv_g1(h2)=", h5, "which is\n equal to the orignal h1.")

######
# Plotting

spot_color = gplt.crimson

g1_h1 = g1*h1
g2_h1 = g2*h1
g1_h2 = g1*h2

h2scale = G.element([h2[0], 0])
h2scale_g1 = h2scale*g1

h3scale = G.element([h3[0], 0])
h3scale_g2 = h3scale * g2

h2shift = G.element([1, h2[1]])
g1_h2shift = g1 * h2shift

h4scale = G.element([h4[0], 0])
h4scale_g1 = h3scale * g1

ax_g1 = plt.subplot(1, 3, 1)
ax = ax_g1
ax.scatter(*g1, edgecolor=spot_color, facecolor='white', label='$g_{1}$')
ax.scatter(*g1_h1, edgecolor='black', facecolor=spot_color, label='$g_{2}$')
ax.scatter(1, 0, edgecolor='black', facecolor='black')
ax.scatter(0, 0, edgecolor='black', facecolor='white', s=20, clip_on=False, zorder=3)
ax.set_xlim(0, 3)
ax.set_ylim(-1, 5)
ax.margins(x=.5)
#ax.set_xticks([0, 1, 2, 3, 4])
#ax.set_yticks([0, 1, 2, 3, 4])
ax.set_aspect('equal')
ax.grid(True)
ax.set_axisbelow(True)
ax.axhline(0, color='black')
ax.arrow(g1[0], g1[1], .8*(g1_h1[0] - g1[0]), .8*(g1_h1[1] - g1[1]), color='black', zorder=.9, length_includes_head=True, width=.01, head_width=0.2, overhang=.5)
ax.arrow(g1[0], g1[1], (h2scale_g1[0] - g1[0]), (h2scale_g1[1] - g1[1]), color=spot_color, zorder=.9, length_includes_head=True, width=.01, head_width=0.2, overhang=.5)
ax.arrow(h2scale_g1[0], h2scale_g1[1], .8*(g1_h1[0] - h2scale_g1[0]), .8*(g1_h1[1] - h2scale_g1[1]), color=spot_color, zorder=.9, length_includes_head=True, width=.01, head_width=0.2, overhang=.5)
ax.plot([0, h2scale_g1[0]], [0, h2scale_g1[1]], color='black', linewidth=.5, linestyle='dashed', zorder=-1)


ax_g2 = plt.subplot(1, 3, 2)
ax = ax_g2
ax.scatter(*g2, edgecolor=spot_color, facecolor='white', label='$g_{1}$')
ax.scatter(*g2_h1, edgecolor='black', facecolor=spot_color, label='$g_{2}$')
ax.scatter(1, 0, edgecolor='black', facecolor='black')
ax.scatter(0, 0, edgecolor='black', facecolor='white', s=20, clip_on=False, zorder=3)
ax.set_xlim(0, 3)
ax.set_ylim(-1, 5)
ax.margins(x=.5)
#ax.set_xticks([0, 1, 2, 3, 4])
#ax.set_yticks([0, 1, 2, 3, 4])
ax.set_aspect('equal')
ax.grid(True)
ax.set_axisbelow(True)
ax.axhline(0, color='black')
ax.arrow(g2[0], g2[1], .8*(g2_h1[0] - g2[0]), .8*(g2_h1[1] - g2[1]), color='black', zorder=.9, length_includes_head=True, width=.01, head_width=0.2, overhang=.5)
ax.arrow(g2[0], g2[1], (h3scale_g2[0] - g2[0]), (h3scale_g2[1] - g2[1]), color=spot_color, zorder=.9, length_includes_head=True, width=.01, head_width=0.2, overhang=.5)
ax.arrow(h3scale_g2[0], h3scale_g2[1], .9 * (g2_h1[0] - h3scale_g2[0]), .9 * (g2_h1[1] - h3scale_g2[1]), color=spot_color, zorder=.9, length_includes_head=True, width=.01, head_width=0.2, overhang=.5)
ax.plot([0, h3scale_g2[0]], [0, h3scale_g2[1]], color='black', linewidth=.5, linestyle='dashed', zorder=-1)

ax_h4 = plt.subplot(1, 3, 3)
ax = ax_h4
ax.scatter(*g1, edgecolor=spot_color, facecolor='white', label='$g_{1}$')
ax.scatter(*g1_h2, edgecolor='black', facecolor=spot_color, label='$g_{2}$')
ax.scatter(1,0, edgecolor='black', facecolor='black')
ax.scatter(0, 0, edgecolor='black', facecolor='white', s=20, clip_on=False, zorder=3)
ax.set_xlim(0, 3)
ax.set_ylim(-1, 5)
ax.margins(x=.5)
#ax.set_xticks([0, 1, 2, 3, 4])
#ax.set_yticks([0, 1, 2, 3, 4])
ax.set_aspect('equal')
ax.grid(True)
ax.set_axisbelow(True)
ax.axhline(0, color='black')
ax.arrow(g1[0], g1[1], (g1_h2shift[0] - g1[0]), (g1_h2shift[1] - g1[1]), color='black', zorder=.9, length_includes_head=True, width=.01, head_width=0.2, overhang=.5)
ax.arrow(g1_h2shift[0], g1_h2shift[1], .85*(g1_h2[0] - g1_h2shift[0]), .85*(g1_h2[1] - g1_h2shift[1]), color='black', zorder=.9, length_includes_head=True, width=.01, head_width=0.2, overhang=.5)
ax.arrow(g1[0], g1[1], (h4scale_g1[0] - g1[0]), (h4scale_g1[1] - g1[1]), color=spot_color, zorder=.9, length_includes_head=True, width=.01, head_width=0.2, overhang=.5)
ax.arrow(h4scale_g1[0], h4scale_g1[1], .95 * (g1_h2[0] - h4scale_g1[0]), .95 * (g1_h2[1] - h4scale_g1[1]), color=spot_color, zorder=.9, length_includes_head=True, width=.01, head_width=0.2, overhang=.5)
ax.plot([0, h2scale_g1[0]], [0, h2scale_g1[1]], color='black', linewidth=.5, linestyle='dashed', zorder=-1)


plt.show()
