import numpy as np
import matplotlib.pyplot as plt
from geomotion import plottingfunctions as gplt
from geomotion import utilityfunctions as ut
from E100_Construct_R2 import R2  # Get the R2 manifold constructed previously


# Make things print nicely
np.set_printoptions(precision=2)

# Make the working manifold for this problem R2
Q = R2

# Construct a point on the manifold
q = Q.element([3, 3], 0)
print("Initial configuration in Cartesian coordinates is " + str(q.value))

# Convert the point to polar coordinates
q_polar = q.transition(1)
print("Configuration in polar coordinates ", q_polar.value)

# Return the point to Cartesian coordinates
q_cartesian = q_polar.transition(0)
print("Configuration back in Cartesian coordinates " + str(q_cartesian.value))



##############
# Plot the calculated terms
spot_color=gplt.crimson

# Original values
ax_orig = plt.subplot(3, 3, 1)
ax_orig.scatter(q.value[0], q.value[1], color=spot_color)
ax_orig.set_xlim(-.5, 3.5)
ax_orig.set_ylim(-.5, 3.5)
ax_orig.set_xticks([0, 1, 2, 3])
ax_orig.set_yticks([0, 1, 2, 3])
ax_orig.set_aspect('equal')
ax_orig.grid(True)
ax_orig.axhline(0, color='black')
ax_orig.axvline(0, color='black')

# Polar equivalents
ax_polar = plt.subplot(3, 3, 5, projection='polar')
ax_polar.scatter(q_polar.value[1], q_polar.value[0], color=spot_color)
ax_polar.set_rlim(0, 5)
ut.convert_polar_xticks_to_radians(ax_polar)

# Values returned to original coordinates
ax_returned = plt.subplot(3, 3, 9)
ax_returned.scatter(q_cartesian.value[0], q_cartesian.value[1], color=spot_color)
ax_returned.set_xlim(-.5, 3.5)
ax_returned.set_ylim(-.5, 3.5)
ax_returned.set_xticks([0, 1, 2, 3])
ax_returned.set_yticks([0, 1, 2, 3])
ax_returned.set_aspect('equal')
ax_returned.grid(True)
ax_returned.axhline(0, color='black')
ax_returned.axvline(0, color='black')

plt.show()