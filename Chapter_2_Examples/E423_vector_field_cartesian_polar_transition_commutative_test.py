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
X_outward_xy = tb.TangentVectorField(v_outward_xy, Q)

# Build a grid over which to evaluate the vector field
grid_xy = ut.meshgrid_array(np.linspace(-2, 2, 5), np.linspace(-2, 2, 5))

# Evaluate the vector field on the grid
vector_grid = X_outward_xy.grid_evaluate_vector_field(grid_xy)
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

# Transition the vector field into polar coordinates
X_outward_rt = X_outward_xy.transition(1)

# Construct a grid of polar coordinates
grid_rt = ut.meshgrid_array([.5, 1, 2], [0, np.pi / 2, np.pi])

# Evaluate the polar-coordinate-expressed field on the polar grid
vector_grid_rt = X_outward_rt.grid_evaluate_vector_field(grid_rt)

print("X_outward_rt([2, 0]) = ", X_outward_rt([2, 0]))
print("X_outward_rt.grid_evaluate_vector_field([2, 0] = ",
      X_outward_rt.grid_evaluate_vector_field(ut.GridArray([[2], [0]], 1)))
print("X_outward_rt([2, pi/2]) = ", X_outward_rt([2, np.pi / 2]))
print("X_outward_rt.grid_evaluate_vector_field([2, pi/2] = ",
      X_outward_rt.grid_evaluate_vector_field(ut.GridArray([[2], [np.pi / 2]], 1)))

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
vector_set_xy = X_outward_xy.grid_evaluate_vector_field(grid_xy, 0, None, None, 'set')
vector_set_rt = X_outward_rt.grid_evaluate_vector_field(grid_rt, 0, None, None, 'set')

# Transition the polar vectors back to Cartesian coordinates
vector_set_rtxy = vector_set_rt.transition(0, 'match')

# Extract the grids from the set-evaluated vector fields
vs_xy_v, vs_xy_c = vector_set_xy.grid
vs_rt_v, vs_rt_c = vector_set_rt.grid
vs_rtxy_v, vs_rtxy_c = vector_set_rtxy.grid

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
ax.quiver(c_grid[0], c_grid[1], v_grid[0], v_grid[1], scale=5)
ax.set_aspect('equal')
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_title("Polar from set")

#
# ax = plt.subplot(2, 2, 4)
# c_grid = vs_rtxy_c
# v_grid = vs_rtxy_v
# ax.quiver(c_grid[0], c_grid[1], v_grid[0], v_grid[1], scale=20, linewidth=10)
# ax.set_aspect('equal')
# ax.set_xlim(-4, 4)
# ax.set_ylim(-4, 4)


plt.show()
