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
q1 = Q.element([3, 3], 0)
q2 = Q.element([3, 0], 0)
q3 = Q.element([1, -1], 0)
q4 = Q.element([0, 2], 0)

# Collect these points into a set
q_set = md.ManifoldElementSet([[q1, q2],
                               [q3, q4]])
print("Initial set in Cartesian coordinates is \n", q_set[0][0].value, "  ", q_set[0][1].value,
      "\n", q_set[1][0].value, "  ", q_set[1][1].value)

# Transition the set into polar coordinates
q_set_polar = q_set.transition(1)

#
print("Set in Polar coordinates is \n", q_set_polar[0][0].value, "  ", q_set_polar[0][1].value,
      "\n", q_set_polar[1][0].value, "  ", q_set_polar[1][1].value)

q_set_cartesian = q_set_polar.transition(0)

print("Set back in Cartesian coordinates ", q_set_cartesian[0][0].value, " and ", q_set_cartesian[0][1].value,
      "\n and ", q_set_cartesian[1][0].value, " and ", q_set_cartesian[1][1].value)

##############
# Plot the calculated terms
spot_color = gplt.crimson

# Original values
ax_orig = plt.subplot(2, 2, 1)
ax_orig.scatter(q_set.grid[0], q_set.grid[1], color=spot_color)
ax_orig.set_xlim(-.5, 5)
ax_orig.set_ylim(-1.5, 4)
ax_orig.set_xticks([0, 1, 2, 3, 4])
ax_orig.set_yticks([-1, 0, 1, 2, 3])
ax_orig.set_aspect('equal')
ax_orig.grid(True)
ax_orig.axhline(0, color='black')
ax_orig.axvline(0, color='black')

# Polar equivalents
ax_polar = plt.subplot(2, 2, 4, projection='polar')
ax_polar.scatter(q_set_polar.grid[1], q_set_polar.grid[0], color=spot_color)
ax_polar.set_rlim(0, 6)
ut.convert_polar_xticks_to_radians(ax_polar)

plt.show()
