# Make things print nicely
import numpy as np

np.set_printoptions(precision=2)

from geomotion import manifold as md
from geomotion import utilityfunctions as ut

# Get the R2 manifold as constructed previously
from E100_Construct_R2 import R2

# Make the working manifold for this problem R2
Q = R2

# Construct a pair of points on the manifold
q1 = [3, 3]
q2 = [3, 0]
q3 = [1, -1]
q4 = [0, 2]

# Collect these points into a set
q_numeric = ut.GridArray([[q1, q2, q1], [q3, q4, q2]], 2)
q_set = md.ManifoldElementSet(Q, q_numeric)

#print("Generated manifold elements are: ", q_set[0].value, " and ", q_set[1].value)



print("Initial set in Cartesian coordinates is ", str(q_set[0][0].value), " and ", str(q_set[0][1].value),
      "\n and ", str(q_set[1][0].value), " and ", str(q_set[1][1].value))

# Transition the set into polar coordinates
q_set_polar = q_set.transition(1)

#
print("Set in Polar coordinates is ", str(q_set_polar.value[0][0].value), " and ", str(q_set_polar.value[0][1].value),
      "\n and ", str(q_set_polar.value[1][0].value), " and ", str(q_set_polar.value[1][1].value))

q_set_cartesian = q_set_polar.transition(0)

print("Set back in Cartesian coordinates ", str(q_set_cartesian.value[0][0].value), " and ",
      str(q_set_cartesian.value[0][1].value),
      "\n and ", str(q_set_cartesian.value[1][0].value), " and ", str(q_set_cartesian.value[1][1].value))
