import numpy as np
from geomotion import manifold as md, diffmanifold as tb, utilityfunctions as ut
from S400_Construct_R2 import R2
from matplotlib import pyplot as plt
np.set_printoptions(precision=2)  # Make things print nicely

Q = R2


# Define a vector field function that points outward everywhere
def v_outward_xy(q):
    v = np.array([q[0], q[1]])
    return v


# Use the vector field function to construct a vector field that points outward, based on a Cartesian definition
X_outward = tb.TangentVectorField(Q, v_outward_xy, 0, 0)

# Build a grid over which to evaluate the vector field
grid_xy = ut.meshgrid_array(np.linspace(-2, 2, 5), np.linspace(-2, 2, 5))

# Turn this grid into a set of points in the manifold
cartesian_points = Q.element_set(grid_xy, 0, 'component')

# Evaluate the vector field on the grid of points
vector_grid = X_outward(cartesian_points).grid[1]
print("The Cartesian components of the outward field are the same as the underlying Cartesian coordinates: \n",
      vector_grid,
      "\n")



# Construct a grid of polar coordinates
grid_rt = ut.meshgrid_array([.5, 1, 2], [0, np.pi / 2, np.pi])

# Turn this grid into a set of points in the manifold
polar_points = Q.element_set(grid_rt, 1, 'component')

# Evaluate the field on the polar grid, and transition the output to polar coordinates
vector_grid_rt = X_outward(polar_points).transition(1).grid

print("The polar components of the outward field are all in the radial direction: \n", vector_grid_rt)

# Adding vector fields
X_doubled = X_outward + X_outward
vector_grid_doubled = X_doubled(cartesian_points).grid[1]

print("Adding vector fields expressed in different coordinates produces a vector field in the first field's "
      "coordinates: \n", vector_grid_doubled)

# Scalar multiplying vector fields
X_tripled = X_outward * 3
vector_grid_tripled = X_tripled(cartesian_points).grid[1]

print("Multiplying a scalar by a vector field scales the output: \n", vector_grid_tripled)

# Dividing a vector field by a scalar
X_halved = X_outward / 2
vector_grid_halved = X_halved(cartesian_points).grid[1]

print("Dividing a vector field by a scalar scales down the value \n", vector_grid_halved, "\n")

# Test that transitioning the vector fields and then evaluating them is the same as evaluating
# them as sets and then transitioning them

# Evaluate the Cartesian and polar expressions of the vector fields as sets
vector_set_xy = X_outward(cartesian_points)

X_outward_polar = X_outward.transition_output(1, 1)
vector_set_rt = X_outward(polar_points)

# Transition the polar set to Cartesian form
vector_set_rtxy = vector_set_rt.transition(0, 'match')

# Extract the grids from the set-evaluated vector fields
vs_xy_v, vs_xy_c = vector_set_xy.grid
vs_rtxy_v, vs_rtxy_c = vector_set_rtxy.grid

print("Shape of vector grids is ", vs_rtxy_v.shape)
print(vs_xy_v[0], "\n", vs_rtxy_v[0])#, "\n", vector_grid[0], "\n", vector_grid_rt[0])

ax = plt.subplot(2, 2, 1)
c_grid = vs_xy_c
v_grid = vs_xy_v
ax.quiver(*c_grid, *v_grid, scale=20, linewidth=10)
ax.set_aspect('equal')
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)

ax = plt.subplot(2, 2, 2)
c_grid = grid_xy
v_grid = vector_grid
ax.quiver(*c_grid, *v_grid, scale=20, linewidth=10)
ax.set_aspect('equal')
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)


ax = plt.subplot(2, 2, 4)
c_grid = vs_rtxy_c
v_grid = vs_rtxy_v
ax.quiver(*c_grid, *v_grid, scale=20, linewidth=10)
ax.set_aspect('equal')
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)

plt.show()

# # Test for equality
# print("Transition values successfully commute: ",
#       "\n", np.isclose(vector_set_rt.value[0][0].value, vector_set_xy.transition(1, 'match')[0][0].value), "   ",
#       np.isclose(vector_set_rt.value[0][1].value, vector_set_xy.transition(1, 'match')[0][1].value), "   ",
#       np.isclose(vector_set_rt.value[0][2].value, vector_set_xy.transition(1, 'match')[0][2].value),
#       "\n", np.isclose(vector_set_rt.value[1][0].value, vector_set_xy.transition(1, 'match')[1][0].value), "   ",
#       np.isclose(vector_set_rt.value[1][1].value, vector_set_xy.transition(1, 'match')[1][1].value), "   ",
#       np.isclose(vector_set_rt.value[1][2].value, vector_set_xy.transition(1, 'match')[1][2].value),)
#
# print("Transition values being tested: ",
#       "\n", vector_set_rt.value[0][0].value, " ", vector_set_xy.transition(1, 'match')[0][0].value, "   ",
#       vector_set_rt.value[0][1].value, " ", vector_set_xy.transition(1, 'match')[0][1].value, "   ",
#       vector_set_rt.value[0][2].value, " ", vector_set_xy.transition(1, 'match')[0][2].value,
#       "\n", vector_set_rt.value[1][0].value, " ", vector_set_xy.transition(1, 'match')[1][0].value, "   ",
#       vector_set_rt.value[1][1].value, " ", vector_set_xy.transition(1, 'match')[1][1].value, "   ",
#       vector_set_rt.value[1][2].value, " ", vector_set_xy.transition(1, 'match')[1][2].value,)
