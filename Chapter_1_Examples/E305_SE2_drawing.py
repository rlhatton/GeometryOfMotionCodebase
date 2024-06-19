from S300_Construct_SE2 import SE2
from S300_Construct_SE2 import RigidBody
from S300_Construct_SE2 import cornered_triangle
from geomotion import plottingfunctions as gplt
from matplotlib import pyplot as plt
import numpy as np

spot_color = gplt.crimson


# Initial positions of elements
e = SE2.identity_element()
g = SE2.element([2, 1, np.pi / 2])
h = SE2.element([0, -1, -np.pi / 4])

# Make three rigid bodies from the elements
a = RigidBody(cornered_triangle(.25, spot_color), e)
b = RigidBody(cornered_triangle(.25, spot_color), g)
c = RigidBody(cornered_triangle(.25, spot_color), h)

# Elements for second plot
gh = g*h

d = RigidBody(cornered_triangle(.25, spot_color), gh)

# Elements for third plot
g_theta = SE2.element([0, 0, g[2]])
g_theta_h = g_theta * h

f = RigidBody(cornered_triangle(.25, spot_color), g_theta_h)

# Plotting commands
ax = plt.subplot(1, 3, 1)
a.draw(ax)
b.draw(ax)
c.draw(ax)
ax.set_xlim(-1, 4)
ax.set_ylim(-2, 3)
ax.set_aspect('equal')
ax.axhline(0, color='black', zorder=-1)
ax.axvline(0, color='black', zorder=-1)

ax = plt.subplot(1, 3, 2)
b.draw(ax)
d.draw(ax)
ax.set_xlim(-1, 4)
ax.set_ylim(-2, 3)
ax.arrow(g[0], g[1], -.75, 0, length_includes_head=True,  width=.01, head_width=0.2, overhang=.5)
ax.arrow(g[0], g[1], 0, .75, length_includes_head=True,  width=.01, head_width=0.2, overhang=.5)
ax.set_aspect('equal')
ax.axhline(0, color='black', zorder=-1)
ax.axvline(0, color='black', zorder=-1)

ax = plt.subplot(1, 3, 3)
c.draw(ax)
f.draw(ax)
d.draw(ax)
ax.set_xlim(-1, 4)
ax.set_ylim(-2, 3)
ax.set_aspect('equal')
circle = plt.Circle([0, 0], 1, edgecolor='black', facecolor='none', linestyle='dashed')
ax.add_artist(circle)
ax.arrow(g_theta_h[0], g_theta_h[1], .9*(gh[0]-g_theta_h[0]), .9*(gh[1]-g_theta_h[1]), zorder=0, length_includes_head=True,  width=.01, head_width=0.2, overhang=.5)
ax.axhline(0, color='black', zorder=-1)
ax.axvline(0, color='black', zorder=-1)

plt.show()
