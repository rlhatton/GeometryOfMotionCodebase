import numpy as np
from geomotion import manifold as md, diffmanifold as tb, utilityfunctions as ut
from S400_Construct_R2 import R2

Q = R2


# Define a vector field function that points outward everywhere
def v_outward_xy(q):
    v = np.array([q[0], q[1]])
    return v

# Use the vector field function to construct a vector field
X_outward_xy = tb.TangentVectorField(v_outward_xy, Q)

# Integrating a flow on the vector field
sol = X_outward_xy.integrate([0, 1], [1, 2])

print("Flow solution is: \n", sol.y, "\n")

# Integrating a flow on the polar vector field
sol = X_outward_rt.integrate([0, 1], [1, np.pi/3])

print("Flow solution in polar coordinates is: \n", sol.y)

# Final point integration of a flow on the cartesian field
finalpoint = X_outward_xy.integrate([0, 1], [1, 2],'final')

print("Flow solution in cartesian coordinates is: \n", finalpoint.value)

# Exponentiation on flow
exppoint = X_outward_xy.exp([1, 2])
print("Exponential in cartesian coordinates is: \n", exppoint.value)
