#! /usr/bin/python3
import numpy as np
from S200_Construct_R2 import R2

# Make things print nicely
np.set_printoptions(precision=2)


# Take the manifold as R2 with a differentiable structure
Q = R2

# Construct an element of R2
q1 = Q.element([2, 0], 0)

# Construct a vector at q; note the column structure
v1 = Q.vector([[0], [1]], q1)

# Convert the vector to the polar basis and polar coordinates
v1_polar = v1.transition(1)

print("Polar expression of Cartesian \n", v1, "\n at ", q1, " is \n", v1_polar, "\n at ", v1_polar.configuration)

# Construct a second element of R2
q2 = Q.element([1, 1], 0)

# Construct a vector at this point
v2 = Q.vector([[1], [1]], q2)

# Convert to polar coordinates and basis
v2_polar = v2.transition(1)

print("\nPolar expression of Cartesian \n", v2, "\n at ", q2, " is \n", v2_polar, "\n at ", v2_polar.configuration)

# Generate a vector in a polar basis, but that retains its Cartesian configuration specification
v2_polar_cartesian_chart = v2.transition(1, 'keep')

print("\n\nOutput vector has basis ", v2_polar_cartesian_chart.current_basis,
      " and configuration chart ", v2_polar_cartesian_chart.configuration.current_chart)
