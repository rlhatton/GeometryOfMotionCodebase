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
X_saddle = tb.TangentVectorField(Q, v_saddle)

# Integrate a flow on the vector field
start1 = 0
end1 = 2.5
sol1 = X_saddle.integrate([start1, end1], [-1.5, 1.8])

print("Flow solution is: \n", sol1.y, "\n")

# Integrate a second flow on the saddle vector field
start2 = 0
end2 = 2.5
sol2 = X_saddle.integrate([start2, end2], [-1.8, 1.5])

print("Flow solution in polar coordinates is: \n", sol2.y)

# Build a grid over which to evaluate the vector field
grid_xy = ut.meshgrid_array(np.linspace(-2, 2, 5), np.linspace(-2, 2, 5))

# Evaluate the saddle vector field on its grid
vector_grid = X_saddle.grid(grid_xy)

# Evaluate solutions at a dense grid of time points
traj1 = sol1.sol(np.linspace(start1, end1))
traj2 = sol2.sol(np.linspace(start2, end2))

ax = plt.subplot(1, 2, 1)
c_grid = grid_xy
v_grid = vector_grid
ax.quiver(c_grid[0], c_grid[1], v_grid[0], v_grid[1], scale=20, linewidth=10)
ax.plot(traj1[0], traj1[1], color=spot_color)
plt.arrow(traj1[0][-1], traj1[1][-1], traj1[0][-1]-traj1[0][-2], traj1[1][-1]-traj1[1][-2], shape='full', lw=0,
          length_includes_head=True, head_width=.25, color=spot_color)
ax.plot(traj2[0], traj2[1], color=spot_color)
plt.arrow(traj2[0][-1], traj2[1][-1], traj2[0][-1]-traj2[0][-2], traj2[1][-1]-traj2[1][-2], shape='full', lw=0,
          length_includes_head=True, head_width=.25, color=spot_color)
ax.set_aspect('equal')
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_title("Saddle field flows")


# Final point integration of a flow on the cartesian field
finalpoint = X_saddle.integrate([0, 1], [1, -1.8], 'final')

print("Flow solution in cartesian coordinates is: \n", finalpoint.value)

# Exponentiation on flow
exppoint = X_saddle.exp([1, -1.8])
print("Exponential in cartesian coordinates is: \n", exppoint.value)

plt.show()