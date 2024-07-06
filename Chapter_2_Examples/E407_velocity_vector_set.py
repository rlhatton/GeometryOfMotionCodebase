#! /usr/bin/python3
import numpy as np
from S400_Construct_R2 import R2
from geomotion import diffmanifold as dm
from geomotion import utilityfunctions as ut
import matplotlib.pyplot as plt

# Make things print nicely
np.set_printoptions(precision=2)


# Take the manifold as R2 with a differentiable structure
Q = R2

# Construct several elements of R2
q11 = Q.element([2, 0], 0)
q12 = Q.element([0, 3], 0)
q21 = Q.element([-1, 0], 0)
q22 = Q.element([1, -1], 0)

# Construct vectors at the configurations
v11 = Q.vector(q11, [0, 9])
v12 = Q.vector(q12, [5, 3])
v21 = Q.vector(q21, [8, 7])
v22 = Q.vector(q22, [4, 2])

# Construct a TangentVectorSet from the vectors
v_set1 = dm.TangentVectorSet([[v11, v12], [v21, v22]])
print("Successfully created TangentVectorSet \n", v_set1, "from TangentVectors")

# Construct a TangentVectorSet from component-wise grids of vector and configuration components
vector_components_c = ut.GridArray([[[0, 5], [8, 4]], [[9, 3], [7, 2]]], 1)
config_components_c = ut.GridArray([[[2, 0], [-1, 1]], [[0, 3], [0, -1]]], 1)

print("Shape of vector_components_c is ", vector_components_c.shape)
print("Shape of vector_components_c everse is ", vector_components_c.everse.shape)

v_grid1, c_grid1 = v_set1.grid
print(v_grid1[0])
print(c_grid1.shape)

ax = plt.subplot(1, 3, 1)
ax.quiver(c_grid1[0], c_grid1[1], v_grid1[0], v_grid1[1])
ax.set_aspect('equal')

v_set2 = dm.TangentVectorSet(Q, config_components_c, vector_components_c, 0, 0, 'component')
print("Successfully created TangentVectorSet \n", v_set2, "\nfrom component-wise grids")

# Construct a TangentVectorSet from element-wise grids of vector and configuration components
vector_components_e = ut.GridArray(vector_components_c.everse, 2)
config_components_e = ut.GridArray(config_components_c.everse, 2)

print("Shape of vector_components_e is ", vector_components_e.shape)
print("Shape of config_components_e is ", config_components_e.shape)


v_set3 = dm.TangentVectorSet(Q, config_components_e, vector_components_e, 0, 0, 'element')
print("Successfully created TangentVectorSet \n", v_set3, "\nfrom element-wise grids")

# Construct a TangentVectorSet from an element-wise grid of vector components and a single
# ManifoldElement

v_set4 = dm.TangentVectorSet(Q, q11, vector_components_e, 0, 0, 'element')
print("Successfully created TangentVectorSet \n", v_set4, "\nfrom element-wise grid and single manifold element")

# Construct a TangentVectorSet from an element-wise grid of vector components and a single
# ndarray of manifold components

v_set5 = dm.TangentVectorSet(Q, np.array([0, 1]), vector_components_e, 0, 0, 'element')
print("Successfully created TangentVectorSet \n", v_set5,
      "\nfrom element-wise grid and single ndarray of manifold components")

# Construct a TangentVectorSet from an element-wise grid of vector components and a single
# list of manifold components

v_set6 = dm.TangentVectorSet(Q, vector_components_e, [0, 1], 0, 0, 'element')
print("Successfully created TangentVectorSet \n", v_set6,
      "\nfrom element-wise grid and single list of manifold components")

v_set7 = v_set6.transition(1, 'match')
print("Successfully transitioned TangentVectorSet \n", v_set6,
      "\ninto ", v_set7)

v_set8 = v_set6 * 3
print("Successfully tripled TangentVectorSet \n", v_set6,
      "\ninto ", v_set8)

v_set9 = v_set6 + v_set6
print("Successfully added TangentVectorSet \n", v_set6, "to itself, resulting"
      "\nin ", v_set9)

# v_set10 = v_set1 + v11

plt.show()
