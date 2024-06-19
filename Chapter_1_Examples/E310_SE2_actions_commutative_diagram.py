from S300_Construct_SE2 import SE2
from S300_Construct_SE2 import RigidBody
from S300_Construct_SE2 import cornered_triangle
from geomotion import plottingfunctions as gplt
from matplotlib import pyplot as plt
import numpy as np

spot_color = gplt.crimson

# Position of starting element
g_initial = SE2.element([1, 2, np.pi / 6])

# Left and right actions to be applied to elements
g_delta_left = SE2.element([-1.5, -1, -np.pi / 3])
h_delta_right = SE2.element([1, -1, np.pi / 2])



print("Moving an object at ", g_initial, " by a right action of ", h_delta_right,
      "\nand then a left action of ", g_delta_left, " places it at ", g_delta_left * (g_initial * h_delta_right), ","
                                                                                                                  "\nwhich is the same position as is reached by first moving it by a left action of ",
      g_delta_left,
      "\n and then a right action of ", h_delta_right, ", ", (g_delta_left * g_initial) * h_delta_right, ".")

####
# Plotting stuff

gd_theta = SE2.element([0, 0, g_delta_left[2]])
gi_h = g_initial * h_delta_right
gd_theta_gi = gd_theta * g_initial
gd_theta_gi_h = gd_theta * (g_initial * h_delta_right)
gd_gi = g_delta_left * g_initial
gd_gi_h = g_delta_left * (g_initial * h_delta_right)

# Make rigid bodies from the elements
a = RigidBody(cornered_triangle(.25, spot_color), g_initial)
b = RigidBody(cornered_triangle(.25, spot_color), g_initial * h_delta_right)
c = RigidBody(cornered_triangle(.25, spot_color), g_delta_left * g_initial)
cp = RigidBody(cornered_triangle(.25, 'grey'), gd_theta * g_initial)
d = RigidBody(cornered_triangle(.25, spot_color), g_delta_left * (g_initial * h_delta_right))
e = RigidBody(cornered_triangle(.25, 'grey'), gd_theta_gi_h)

# Plotting commands
ax = plt.subplot(1, 1, 1)
a.draw(ax)
b.draw(ax)
c.draw(ax)
cp.draw(ax)
d.draw(ax)
e.draw(ax)
ax.set_xlim(-1, 4)
ax.set_ylim(-4, 3)
circle = plt.Circle([0, 0], np.sqrt(g_initial[0]*g_initial[0] + g_initial[1]*g_initial[1]), edgecolor='black', facecolor='none', linestyle='dashed')
ax.add_artist(circle)
circle = plt.Circle([0, 0], np.sqrt(gi_h[0]*gi_h[0] + gi_h[1]*gi_h[1]), edgecolor='black', facecolor='none', linestyle='dashed')
ax.add_artist(circle)
ax.set_aspect('equal')
ax.axhline(0, color='black', zorder=-1)
ax.axvline(0, color='black', zorder=-1)
circle = plt.Circle([0, 0], np.sqrt(g_initial[0]*g_initial[0] + g_initial[1]*g_initial[1]), edgecolor='black', facecolor='none', linestyle='dashed')
ax.add_artist(circle)
circle = plt.Circle([0, 0], np.sqrt(gi_h[0]*gi_h[0] + gi_h[1]*gi_h[1]), edgecolor='black', facecolor='none', linestyle='dashed')
ax.add_artist(circle)
ax.arrow(gd_theta_gi[0], gd_theta_gi[1], .9*(gd_gi[0]-gd_theta_gi[0]), .9*(gd_gi[1]-gd_theta_gi[1]), zorder=0, length_includes_head=True,  width=.01, head_width=0.2, overhang=.5, facecolor='black')
ax.arrow(gd_theta_gi_h[0], gd_theta_gi_h[1], .9*(gd_gi_h[0]-gd_theta_gi_h[0]), .9*(gd_gi_h[1]-gd_theta_gi_h[1]), zorder=0, length_includes_head=True,  width=.01, head_width=0.2, overhang=.5, color='black')
ax.arrow(g_initial[0], g_initial[1], .9*(gi_h[0]-g_initial[0]), .9*(gi_h[1]-g_initial[1]), zorder=0, length_includes_head=True,  width=.01, head_width=0.2, overhang=.5, color=spot_color)
ax.arrow(gd_gi[0], gd_gi[1], .9*(gd_gi_h[0]-gd_gi[0]), .9*(gd_gi_h[1]-gd_gi[1]), zorder=0, length_includes_head=True,  width=.01, head_width=0.2, overhang=.5, color=spot_color)


plt.show()
