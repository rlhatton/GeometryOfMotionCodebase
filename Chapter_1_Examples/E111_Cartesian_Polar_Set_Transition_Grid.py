# Make things print nicely
import numpy as np
np.set_printoptions(precision=2)

from geomotion import manifold as md

# Get the R2 manifold as constructed previously
from E100_Construct_R2 import R2



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
print("Initial set in Cartesian coordinates is ", str(q_set.value[0][0].value), " and ", str(q_set.value[0][1].value),
      "\n and ", str(q_set.value[1][0].value), " and ", str(q_set.value[1][1].value))

# Transition the set into polar coordinates
q_set_polar = q_set.transition(1)

#
print("Set in Polar coordinates is ", str(q_set_polar.value[0][0].value), " and ", str(q_set_polar.value[0][1].value),
      "\n and ", str(q_set_polar.value[1][0].value), " and ", str(q_set_polar.value[1][1].value))

q_set_cartesian = q_set_polar.transition(0)

print("Set back in Cartesian coordinates ", str(q_set_cartesian.value[0][0].value), " and ", str(q_set_cartesian.value[0][1].value),
      "\n and ", str(q_set_cartesian.value[1][0].value), " and ", str(q_set_cartesian.value[1][1].value))
