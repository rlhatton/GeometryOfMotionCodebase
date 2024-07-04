import numpy as np
from geomotion import manifold as md, diffmanifold as tb, utilityfunctions as ut, plottingfunctions as gplt
from S400_Construct_R2 import R2
from matplotlib import pyplot as plt

# Make things print nicely
np.set_printoptions(precision=2)
spot_color = gplt.crimson

# Make a single-chart 1-dimensional manifold
R1 = tb.DiffManifold([[None]], 1)


# Define a unit-velocity flow along the R1 space
def unit_flow_func(x):
    return 1


# Make a unit-flow vector field
unit_flow_field = tb.TangentVectorField(R1, unit_flow_func)


# Define an immersion from R1 to R2
def immerse_R1_in_R2(s):
    output = [np.sin(s[0])+.5*s[0], np.cos(s[0])]
    return output


immersion_map_R1_into_R2 = md.ManifoldMap(R1, R2, immerse_R1_in_R2)

# Define the differential map associated with the immersion map
diffimmersion_map_R1_into_R2 = tb.DifferentialMap(R1, R2, immersion_map_R1_into_R2)

# Generate sparse and dense sets of points along a curve
s_sparse = ut.GridArray([np.linspace(0, 2*np.pi, 9)], n_outer=1)
s_dense = ut.GridArray([np.linspace(0, 2*np.pi)], n_outer=1)

q_R1_sparse = R1.element_set(s_sparse)
q_R1_dense = R1.element_set(s_dense)

# Map the dense set of points into R2
xy_dense = immersion_map_R1_into_R2(q_R1_dense)

# Map the sparse set of points to unit vectors, and then immerse them
v_sparse = unit_flow_field(s_sparse)
v_xy_sparse = diffimmersion_map_R1_into_R2(v_sparse)


ax = plt.subplot(1, 1, 1)
ax.plot(*xy_dense.grid, color=spot_color)
ax.quiver(*v_xy_sparse.grid[0], *v_xy_sparse.grid[1], scale=10)
ax.set_aspect('equal')

plt.show()