import numpy as np
from geomotion import manifold as md, diffmanifold as tb, utilityfunctions as ut, plottingfunctions as gplt
from S400_Construct_R2 import R2
from matplotlib import pyplot as plt
# Make things print nicely
np.set_printoptions(precision=2)
spot_color = gplt.crimson


# Set the manifold to be R2
Q = R2


# Define a vector field function that is the derivative of a saddle surface
def v_saddle(q):
    v = np.array([q[1], q[0]])
    return v


# Use the vector field function to construct a vector field
X_saddle = tb.TangentVectorField(Q, v_saddle, 0, 0)

# Integrate a flow on the vector field
start1 = 0
end1 = 2.5
q1_initial = Q.element([-1.5, 1.8], 0)
sol1 = X_saddle.integrate([start1, end1], q1_initial)

print("Flow solution is: \n", sol1.y, "\n")

# Integrate a second flow on the saddle vector field
start2 = 0
end2 = 2.5
q2_initial = Q.element([-1.8, 1.5], 0)
sol2 = X_saddle.integrate([start2, end2], q2_initial)

print("Flow solution for second flow is: \n", sol2.y)

# Build a grid over which to evaluate the vector field
grid_xy = ut.meshgrid_array(np.linspace(-2, 2, 5), np.linspace(-2, 2, 5))
cartesian_points = Q.element_set(grid_xy,0)

# Evaluate the saddle vector field on its grid
vector_set = X_saddle(cartesian_points)

# Evaluate solutions at a dense grid of time points
traj1 = sol1.sol(np.linspace(start1, end1))
traj2 = sol2.sol(np.linspace(start2, end2))

ax = plt.subplot(1, 2, 1)
c_grid, v_grid = vector_set.grid
ax.quiver(*c_grid, *v_grid, scale=20, linewidth=10)
ax.plot(*traj1, color=spot_color)
plt.arrow(traj1[0][-1], traj1[1][-1], traj1[0][-1]-traj1[0][-2], traj1[1][-1]-traj1[1][-2], shape='full', lw=0,
          length_includes_head=True, head_width=.25, color=spot_color)
ax.plot(*traj2, color=spot_color)
plt.arrow(traj2[0][-1], traj2[1][-1], traj2[0][-1]-traj2[0][-2], traj2[1][-1]-traj2[1][-2], shape='full', lw=0,
          length_includes_head=True, head_width=.25, color=spot_color)
ax.set_aspect('equal')
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_title("Saddle field flows")


######################################
######################################
# Repeat the above, but using polar coordinates
# (this still cheats a bit in that I haven't defined the
# boundaries over which the chart is valid, but this example would work in a 0-2pi chart)

# Integrate a flow on the vector field
start1 = 0
end1 = 2.5
q1_initial_polar = q1_initial.transition(1)
sol1 = X_saddle.integrate([start1, end1], q1_initial_polar)

print("Flow solution is: \n", sol1.y, "\n")

# Integrate a second flow on the saddle vector field
start2 = 0
end2 = 2.5
q2_initial_polar = q2_initial.transition(1)
sol2 = X_saddle.integrate([start2, end2], q2_initial_polar)

print("Flow solution for second flow is: \n", sol2.y)

# Build a grid over which to evaluate the vector field
grid_rt = ut.meshgrid_array(np.linspace(.5, 3, 5), np.linspace(-np.pi, np.pi, 12))
polar_points = Q.element_set(grid_rt, 1)

# Evaluate the saddle vector field on the polar grid, then transition all the points to cartesian for plotting
vector_set = X_saddle(polar_points).transition(0)

# Evaluate solutions at a dense grid of time points
traj1_raw = sol1.sol(np.linspace(start1, end1))
traj2_raw = sol2.sol(np.linspace(start2, end2))

# Turn the trajectory points into manifold elements in the polar coordinates,
# and transition to Cartesian
traj1_points = Q.element_set(ut.GridArray(traj1_raw, n_inner=1), 1).transition(0)
traj2_points = Q.element_set(ut.GridArray(traj2_raw, n_inner=1), 1).transition(0)

# Get the trajectory numerical values back for plotting
traj1_polar_gen = traj1_points.grid
traj2_polar_gen = traj2_points.grid



ax = plt.subplot(1, 2, 2)
c_grid, v_grid = vector_set.grid
ax.quiver(*c_grid, *v_grid, scale=20, linewidth=10)
ax.plot(*traj1_polar_gen, color=spot_color)
plt.arrow(traj1_polar_gen[0][-1], traj1_polar_gen[1][-1], traj1_polar_gen[0][-1]-traj1_polar_gen[0][-2], traj1_polar_gen[1][-1]-traj1_polar_gen[1][-2], shape='full', lw=0,
          length_includes_head=True, head_width=.25, color=spot_color)
ax.plot(*traj2_polar_gen, color=spot_color)
plt.arrow(traj2_polar_gen[0][-1], traj2_polar_gen[1][-1], traj2_polar_gen[0][-1]-traj2_polar_gen[0][-2], traj2_polar_gen[1][-1]-traj2_polar_gen[1][-2], shape='full', lw=0,
         length_includes_head=True, head_width=.25, color=spot_color)
ax.set_aspect('equal')
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_title("Saddle field flows")






# Final point integration of a flow on the cartesian field
q3 = Q.element([1, -1.8], 0)
finalpoint = X_saddle.integrate([0, 1], q3, 'final')

print("Flow solution in cartesian coordinates is: \n", finalpoint.value)

# Exponentiation on flow
exppoint = X_saddle.exp(q3)
print("Exponential in cartesian coordinates is: \n", exppoint.value)

plt.show()
