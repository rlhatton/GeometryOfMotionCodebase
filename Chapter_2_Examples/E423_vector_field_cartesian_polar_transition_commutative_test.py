import numpy as np
from geomotion import manifold as md, diffmanifold as tb, utilityfunctions as ut
from S400_Construct_R2 import R2
from matplotlib import pyplot as plt

np.set_printoptions(precision=2)  # Make things print nicely

Q = R2


# Define a vector field function that points outward everywhere
def v_outward_xy(q):
    v = [q[0], q[1]]
    return v


# Use the vector field function to construct a vector field
X_outward_xy = tb.TangentVectorField(Q, v_outward_xy)

# Transition the vector field into polar coordinates
X_outward_rt = X_outward_xy.transition(1)

# Build a grid over which to evaluate the vector field
grid_xy = ut.meshgrid_array(np.linspace(-2, 2, 5), np.linspace(-2, 2, 5))

# Transition the grid into polar coordinates (ignore for now that xy=(0,0) does not have a corresponding polar point)
grid_rt = (md.ManifoldElementSet(Q, grid_xy).transition(1)).grid

# Evaluate the Cartesian vector field on its grid
vector_grid = X_outward_xy(grid_xy).grid[1]
print("The Cartesian components of the outward field are the same as the underlying Cartesian coordinates: \n",
      vector_grid,
      "\n")

ax = plt.subplot(3, 2, 1)
c_grid = grid_xy
v_grid = vector_grid
ax.quiver(c_grid[0], c_grid[1], v_grid[0], v_grid[1], scale=20, linewidth=10)
ax.set_aspect('equal')
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_title("Cartesian chart outward")

# Evaluate the polar-coordinate-expressed field on the polar grid
vector_grid_rt = X_outward_rt(grid_rt).grid[1]

print("The polar components of the outward field are all in the radial direction: \n", vector_grid_rt)

ax = plt.subplot(3, 2, 2)
c_grid = grid_rt
v_grid = vector_grid_rt
ax.quiver(c_grid[0], c_grid[1], v_grid[0], v_grid[1], scale=5, units='xy')
ax.set_aspect('equal')
ax.set_xlim(0, 3)
ax.set_ylim(0, 4)
ax.set_title("Polar chart radial")

# Test that transitioning the vector fields and then evaluating them is the same as evaluating
# them as sets and then transitioning them

# Evaluate the Cartesian and polar expressions of the vector fields as sets
vector_set_xy = X_outward_xy(grid_xy)
vector_set_rt = X_outward_rt(grid_rt)

# Extract the grids from the set-evaluated vector fields
vs_xy_c, vs_xy_v = vector_set_xy.grid
vs_rt_c, vs_rt_v = vector_set_rt.grid


# print("Shape of vector grids is ", vs_rtxy_v.shape)
# print(vs_xy_v[0], "\n", vs_rtxy_v[0])  #, "\n", vector_grid[0], "\n", vector_grid_rt[0])

ax = plt.subplot(3, 2, 3)
c_grid = vs_xy_c
v_grid = vs_xy_v
ax.quiver(c_grid[0], c_grid[1], v_grid[0], v_grid[1], scale=20)
ax.set_aspect('equal')
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_title("Cartesian from set")

ax = plt.subplot(3, 2, 4)
c_grid = vs_rt_c
v_grid = vs_rt_v
ax.quiver(c_grid[0], c_grid[1], v_grid[0], v_grid[1], scale=5, units='xy')
ax.set_aspect('equal')
ax.set_xlim(0, 3)
ax.set_ylim(0, 4)
ax.set_title("Polar from set")

# Transition the polar vectors back to Cartesian coordinates
vector_set_rtxy = vector_set_rt.transition(0, 'match')
vs_rtxy_c, vs_rtxy_v = vector_set_rtxy.grid

ax = plt.subplot(3, 2, 5)
c_grid = vs_rtxy_c
v_grid = vs_rtxy_v
ax.quiver(c_grid[0], c_grid[1], v_grid[0], v_grid[1], scale=20)
ax.set_aspect('equal')
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)

# print("\n\n")
# print("Tests of the different polar vector field representations:")
# print("Directly passing inputs to vector field function: X_outward_rt([2, 0]) = \n", X_outward_rt([2, 0]))
# print("Grid evaluating vector field: X_outward_rt.grid_evaluate_vector_field(ut.GridArray([[2], [0]], 1)) = \n",
#       X_outward_rt.grid_evaluate_vector_field(ut.GridArray([[2], [0]], 1)))
# print("Directly passing inputs to vector field function: X_outward_rt([2, pi/2]) = \n", X_outward_rt([2, np.pi / 2]))
# print("Grid evaluating vector field: X_outward_rt.grid_evaluate_vector_field(ut.GridArray([[2], [np.pi / 2]], 1)) = \n",
#       X_outward_rt.grid_evaluate_vector_field(ut.GridArray([[2], [np.pi / 2]], 1)))
# print("\n\n")

plt.show()
