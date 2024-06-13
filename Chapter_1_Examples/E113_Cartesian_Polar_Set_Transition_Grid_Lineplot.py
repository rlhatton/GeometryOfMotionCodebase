import numpy as np
import matplotlib.pyplot as plt
from geomotion import manifold as md
from geomotion import utilityfunctions as ut
from geomotion import plottingfunctions as gplt
from S100_Construct_R2 import R2  # Get the R2 manifold as constructed previously

np.set_printoptions(precision=2)  # Make things print nicely




# Make the working manifold for this problem R2
Q = R2

# Construct the x and y values for a rectangle in the space
q_x = np.concatenate([np.linspace(.5, 2, 20), np.full(20, 2), np.linspace(2, .5, 20), np.full(20, .5)])
q_y = np.concatenate([np.full(20, 0), np.linspace(0, 4, 20), np.full(20, 4), np.linspace(4, 0, 20), ])

# Collect these points into a set
q_numeric = ut.GridArray([q_x, q_y], 1)
q_set_ambient = md.ManifoldElementSet(Q, q_numeric, 0)

q_set_cartesian = q_set_ambient.transition(0)

# Transition the set into polar coordinates
q_set_polar = q_set_cartesian.transition(1)




##############
# Plot the calculated terms
spot_color = gplt.crimson

# Original values
ax_orig = plt.subplot(2, 2, 1)
ax_orig.plot(q_set_cartesian.grid[0], q_set_cartesian.grid[1], color=spot_color)
ax_orig.set_xlim(-.5, 5)
ax_orig.set_ylim(-.5, 5)
ax_orig.set_xticks([0, 1, 2, 3, 4])
ax_orig.set_yticks([0, 1, 2, 3, 4])
ax_orig.set_aspect('equal')
ax_orig.set_axisbelow(True)
ax_orig.grid(True)
ax_orig.axhline(0, color='black', zorder=.75)
ax_orig.axvline(0, color='black', zorder=.75)


# Polar equivalents
ax_polar = plt.subplot(2, 2, 2, projection='polar')
ax_polar.plot(q_set_polar.grid[1], q_set_polar.grid[0], color=spot_color)
ax_polar.set_rlim(0, 6)
ut.convert_polar_xticks_to_radians(ax_polar)

ax_polar = plt.subplot(2, 2, 4)
ax_polar.plot(q_set_polar.grid[1], q_set_polar.grid[0], color=spot_color)
ax_orig.set_xlim(-.5, 5)
ax_orig.set_ylim(-.5, 5)

plt.show()