import numpy as np
import matplotlib.pyplot as plt
from E100_Construct_R2 import R2  # Get the R2 manifold constructed previously
import matplotlib
matplotlib.use('MACOSX')

# Make things print nicely
np.set_printoptions(precision=2)

# Make the working manifold for this problem R2
Q = R2

# Construct a point on the manifold
q = Q.element([3, 3], 0)
print("Initial configuration in Cartesian coordinates is " + str(q.value))
fig, ax = plt.subplots()
ax.plot(q.value, q.value)
ax.grid(True)
plt.show()


q_polar = q.transition(1)

print("Configuration in polar coordinates ", q_polar.value)

q_cartesian = q_polar.transition(0)

print("Configuration back in Cartesian coordinates " + str(q_cartesian.value))
