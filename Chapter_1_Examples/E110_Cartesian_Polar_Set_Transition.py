import numpy as np
import matplotlib.pyplot as plt
from geomotion import manifold as md
from geomotion import utilityfunctions as ut
from geomotion import plottingfunctions as gplt
from S100_Construct_R2 import R2  # Get the R2 manifold as constructed previously

np.set_printoptions(precision=2)  # Make things print nicely

# Make the working manifold for this problem R2
Q = R2

# Construct a pair of points on the manifold
q1 = Q.element([3, 4], 0)
q2 = Q.element([2, 0], 0)

# Collect these points into a set
q_set = md.ManifoldElementSet([q1, q2])
print("Initial set in Cartesian coordinates is ", q_set[0].value, " and ", q_set[1].value)

# Transition the set into polar coordinates
q_set_polar = q_set.transition(1)

#
print("Set in Polar coordinates is ", q_set_polar[0].value, " and ", q_set_polar[1].value)

q_set_cartesian = q_set_polar.transition(0)

print("Set back in Cartesian coordinates ", q_set_cartesian[0].value, " and ", q_set_cartesian[1].value)

q_set_grid = q_set.grid
print("Grid representation of set is ", q_set_grid[0], " and ", q_set_grid[1])


##############
# Plot the calculated terms
spot_color = gplt.crimson

# Original values
ax_orig = plt.subplot(2, 2, 1)
ax = ax_orig
ax.scatter(q_set.grid[0], q_set.grid[1], color=spot_color)
ax.set_xlim(-.5, 5)
ax.set_ylim(-.5, 5)
ax.set_xticks([0, 1, 2, 3, 4])
ax.set_yticks([0, 1, 2, 3, 4])
ax.set_aspect('equal')
ax.grid(True)
ax.axhline(0, color='black')
ax.axvline(0, color='black')

# Polar equivalents
ax_polar = plt.subplot(2, 2, 4, projection='polar')
ax = ax_polar
ax.scatter(q_set_polar.grid[1], q_set_polar.grid[0], color=spot_color)
ax.set_rlim(0, 6)
ut.convert_polar_xticks_to_radians(ax_polar)

plt.show()
