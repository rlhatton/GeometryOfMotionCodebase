import numpy as np
import matplotlib.pyplot as plt
from geomotion import utilityfunctions as ut
from E100_Construct_R2 import R2  # Get the R2 manifold constructed previously


# Make things print nicely
np.set_printoptions(precision=2)

# Make the working manifold for this problem R2
Q = R2

# Construct a point on the manifold
q = Q.element([3, 3], 0)
print("Initial configuration in Cartesian coordinates is " + str(q.value))
ax_orig = plt.subplot(1, 3, 1)
ax_orig.scatter(q.value[0], q.value[1])
ax_orig.set_xlim(-.5, 3.5)
ax_orig.set_ylim(-.5, 3.5)
ax_orig.set_xticks([0, 1, 2, 3])
ax_orig.set_yticks([0, 1, 2, 3])
ax_orig.set_aspect('equal')
ax_orig.grid(True)

q_polar = q.transition(1)
print("Configuration in polar coordinates ", q_polar.value)
ax_polar = plt.subplot(132, projection='polar')
ax_polar.scatter(q_polar.value[1], q_polar.value[0])
ax_polar.set_rlim(0, 5)
ut.convert_polar_xticks_to_radians(ax_polar)

q_cartesian = q_polar.transition(0)
print("Configuration back in Cartesian coordinates " + str(q_cartesian.value))
ax_returned = plt.subplot(1, 3, 3)
ax_returned.scatter(q_cartesian.value[0], q_cartesian.value[1])
ax_returned.set_xlim(-.5, 3.5)
ax_returned.set_ylim(-.5, 3.5)
ax_returned.set_xticks([0, 1, 2, 3])
ax_returned.set_yticks([0, 1, 2, 3])
ax_returned.set_aspect('equal')
ax_returned.grid(True)

plt.show()