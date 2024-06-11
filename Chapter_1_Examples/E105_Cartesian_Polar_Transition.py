# Make things print nicely
import numpy as np
np.set_printoptions(precision=2)

# Get the R2 manifold as constructed previously
from E100_Construct_R2 import R2

# Make the working manifold for this problem R2
Q = R2

# Construct a point on the manifold
q = Q.element([3, 3], 0)

print("Initial configuration in Cartesian coordinates is " + str(q.value))

q_polar = q.transition(1)

print("Configuration in polar coordinates ", q_polar.value)

q_cartesian = q_polar.transition(0)

print("Configuration back in Cartesian coordinates " + str(q_cartesian.value))
