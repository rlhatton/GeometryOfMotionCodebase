import numpy as np
from geomotion import utilityfunctions as ut, plottingfunctions as gplt
from S670_Construct_SE2 import SE2, RigidBody, cornered_triangle
from matplotlib import pyplot as plt

# Make things print nicely
np.set_printoptions(precision=2)
spot_color = gplt.crimson

# Make the working group the SE(2) group
G = SE2

# Make the generating vectors
g_circ_A = G.Lie_alg_vector([0, 0, np.pi/2])
g_circ_B = G.Lie_alg_vector([1, 0, 0])

triangles_ApB=[]
triangles_ApB.append(RigidBody(cornered_triangle(.1, 'black'), G.identity_element()))
triangles_ApB.append(RigidBody(cornered_triangle(.1, spot_color), g_circ_A.exp_R * g_circ_B.exp_R))
# triangles.append(RigidBody(cornered_triangle(.1, 'grey'), (g_circ_A/2).exp_R))
triangles_ApB.append(RigidBody(cornered_triangle(.1, spot_color), (g_circ_A + g_circ_B).exp_R))
# triangles.append(RigidBody(cornered_triangle(.1, 'grey'), ((g_circ_A + g_circ_B)*.5).exp_R))



# Generate a loose set of points from zero to 1
t = np.linspace(0, 1, 50)


###
# get the A-only flow, starting at the origin
g_circ = g_circ_A
start1 = 0
end1 = 1
g_initial = G.identity_element()
sol = G.L_generator(g_circ.value).integrate([start1, end1], g_initial)

# Evaluate the solution trajectory at the time points
traj_A = sol.sol(t)

###
# get the exponential flow, starting at the origin
g_circ = g_circ_A + g_circ_B
start1 = 0
end1 = 1
g_initial = G.identity_element()
sol = G.L_generator(g_circ.value).integrate([start1, end1], g_initial)



# Evaluate the solution trajectory at the time points
traj_ApB = sol.sol(t)

# Now take the Lie bracket and add it to the exponent
g_circ_L_AB = G.Lie_alg_vector([0, np.pi/2, 0])

triangles_ApBpLAB=[]
triangles_ApBpLAB.append(RigidBody(cornered_triangle(.1, 'black'), G.identity_element()))
triangles_ApBpLAB.append(RigidBody(cornered_triangle(.1, spot_color), g_circ_A.exp_R * g_circ_B.exp_R))
# triangles_ApBpLAB.append(RigidBody(cornered_triangle(.1, 'grey'), (g_circ_A/2).exp_R))
triangles_ApBpLAB.append(RigidBody(cornered_triangle(.1, spot_color), (g_circ_A + g_circ_B+(g_circ_L_AB/2)).exp_R))
# triangles_ApBpLAB.append(RigidBody(cornered_triangle(.1, 'grey'), ((g_circ_A + g_circ_B)*.5).exp_R))

###
# get the exponential flow, starting at the origin
g_circ = g_circ_A + g_circ_B + 0.5 * g_circ_L_AB
start1 = 0
end1 = 1
g_initial = G.identity_element()
sol = G.L_generator(g_circ.value).integrate([start1, end1], g_initial)

# Evaluate the solution trajectory at the time points
traj_ApBpLAB = sol.sol(t)

# Now take the next Lie bracket and add it to the exponent
g_circ_L_AmBLAB = G.Lie_alg_vector([np.pi * np.pi /4, 0, 0])

###
# get the exponential flow, starting at the origin
g_circ = g_circ_A + g_circ_B + (g_circ_L_AB/2) + (g_circ_L_AmBLAB/(-12))
start1 = 0
end1 = 1
g_initial = G.identity_element()
sol = G.L_generator(g_circ.value).integrate([start1, end1], g_initial)

# Evaluate the solution trajectory at the time points
traj_L_AmBLAB = sol.sol(t)

ax = plt.subplot(1, 3, 1)
ax.set_xlim(-.25, 1.25)
ax.set_ylim(-.1, 1.5)
ax.set_aspect('equal')
#ax.scatter(1, 0, edgecolor=spot_color, facecolor=spot_color, zorder=-2)
for t in triangles_ApB:
    t.draw(ax)
ax.plot(*traj_A[0:2], color='black', zorder=-3) #, marker='.', markersize=10)
ax.plot(*traj_ApB[0:2], color=spot_color, zorder=-3) #, marker='.', markersize=10)
# ax.plot(*traj_ApBpLAB[0:2], color='grey', zorder=-3) #, marker='.', markersize=10)
# ax.plot(*traj_L_AmBLAB[0:2], color='black', zorder=-3) #, marker='.', markersize=10)
# ax.scatter(0, 0, edgecolor='black', facecolor='white', s=20, clip_on=False, zorder=3)
# ax.axhline(0, color='grey', linewidth=0.5, linestyle="dashed", zorder=0)
# ax.axvline(1, color='grey', linewidth=0.5, linestyle="dashed", zorder=0)
ax.axis('off')

ax = plt.subplot(1, 3, 2)
ax.set_xlim(-.25, 1.25)
ax.set_ylim(-.1, 1.5)
ax.set_aspect('equal')
for t in triangles_ApBpLAB:
    t.draw(ax)
# ax.plot(*traj_A[0:2], color=spot_color, zorder=-3)  # , marker='.', markersize=10)
# ax.scatter(1, 0, edgecolor=spot_color, facecolor=spot_color, zorder=-2)
ax.plot(*traj_ApB[0:2], color='grey', zorder=-3, linestyle='dotted', linewidth=2.25) #, marker='.', markersize=10)
ax.plot(*traj_ApBpLAB[0:2], color=spot_color, zorder=-3) #, marker='.', markersize=10)
# ax.plot(*traj_L_AmBLAB[0:2], color='black', zorder=-3) #, marker='.', markersize=10)
# ax.scatter(0, 0, edgecolor='black', facecolor='white', s=20, clip_on=False, zorder=3)
# ax.axhline(0, color='grey', linewidth=0.5, linestyle="dashed", zorder=0)
# ax.axvline(1, color='grey', linewidth=0.5, linestyle="dashed", zorder=0)
ax.axis('off')

ax = plt.subplot(1, 3, 3)
ax.set_xlim(-.25, 1.25)
ax.set_ylim(-.1, 1.5)
ax.set_aspect('equal')
for t in triangles_ApB[0:2]:
    t.draw(ax)
# ax.scatter(1, 0, edgecolor=spot_color, facecolor=spot_color, zorder=-2)
ax.plot(*traj_ApB[0:2], color='grey', zorder=-3, linestyle='dotted', linewidth=2.25) #, marker='.', markersize=10)
ax.plot(*traj_ApBpLAB[0:2], color='grey', zorder=-3, linestyle='dashed', linewidth=2) #, marker='.', markersize=10)
ax.plot(*traj_L_AmBLAB[0:2], color=spot_color, zorder=-3) #, marker='.', markersize=10)
# ax.scatter(0, 0, edgecolor='black', facecolor='white', s=20, clip_on=False, zorder=3)
# ax.axhline(0, color='grey', linewidth=0.5, linestyle="dashed", zorder=0)
# ax.axvline(1, color='grey', linewidth=0.5, linestyle="dashed", zorder=0)
ax.axis('off')


plt.show()