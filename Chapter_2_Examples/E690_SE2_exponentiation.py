import numpy as np
from geomotion import utilityfunctions as ut, plottingfunctions as gplt
from S670_Construct_SE2 import SE2, RigidBody, cornered_triangle
from matplotlib import pyplot as plt

# Make things print nicely
np.set_printoptions(precision=2)
spot_color = gplt.crimson

# Make the working group the SE(2) group
G = SE2

# Make the generating vector
g_circ = G.Lie_alg_vector([1, 0, np.pi/2])

###
# For display, get the exponential flow, starting at the origin
start1 = 0
end1 = 1
g_initial = G.identity_element()
sol = G.L_generator(g_circ.value).integrate([start1, end1], g_initial)

# Generate a loose set of points from zero to 1
t = np.linspace(0, 1, 50)

# Evaluate the solution trajectory at the time points
traj = sol.sol(t)

ax = plt.subplot(1, 2, 1)
ax.set_xlim(-.25, 1.25)
ax.set_ylim(-1,1)
ax.set_aspect('equal')
# ax.scatter(*g_circ_exp_L, edgecolor=spot_color, facecolor=spot_color, zorder=-2)
ax.plot(*traj[0:2], color=spot_color, zorder=-3) #, marker='.', markersize=10)
# ax.scatter(0, 0, edgecolor='black', facecolor='white', s=20, clip_on=False, zorder=3)
# ax.axhline(0, color='grey', linewidth=0.5, linestyle="dashed", zorder=0)
# ax.axvline(1, color='grey', linewidth=0.5, linestyle="dashed", zorder=0)



plt.show()