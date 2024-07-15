import numpy as np
from geomotion import manifold as md, diffmanifold as tb, utilityfunctions as ut, plottingfunctions as gplt
from S500_Construct_RxRplus import RxRplus
from matplotlib import pyplot as plt

# Make things print nicely
np.set_printoptions(precision=2)
spot_color = gplt.crimson

# Make the working group the scale-shift group
G = RxRplus

# Create a grid in the right-half-plane
grid_xy = ut.meshgrid_array(np.linspace(.5, 2, 4), np.linspace(-1, 1, 5))
# Turn the grid into a set of points in the group
grid_points = RxRplus.element_set(grid_xy)

# Create a pair of groupwise velocity vectors to use in the adjoint demonstration
g_circ_l = G.Lie_alg_vector([1, -0.25])
g_circ_r = G.Lie_alg_vector([1, 0.5])

# Create vectors on the respective generator fields
d_dL = G.L_generator(g_circ_l.value)(grid_points)
d_dR = G.R_generator(g_circ_r.value)(grid_points)

# Create the group element for which the two groupwise velocities are adjoint to each other
g = G.element([1.5, 1])

# Demonstrate that the adjoint of the right velocity is the left, and the adjoint inverse
# of the left velocity is the right
print("The adjoint of g_circ_r=", g_circ_r, " by g=", g, " is ", g.Ad(g_circ_r), " which is equal to g_circ_l.")
print("The adjoint-inverse of g_circ_l=", g_circ_l, " by g=", g, " is ", g.Adinv(g_circ_l), " which is equal to g_circ_r.")
print("The adjoint of g_circ_l=", g_circ_l, " by g_inv=", g.inverse, " is ", g.inverse.Ad(g_circ_l), " which is equal to g_circ_r.")



ax = plt.subplot(1, 3, 1)
c_grid_0, v_grid_0 = d_dL.grid
ax.quiver(*c_grid_0, *v_grid_0, scale=20, color=spot_color)
ax.set_aspect('equal')
ax.set_xlim(0, 2.25)
ax.set_ylim(-1.1, 1.25)
# ax.scatter(1, 0, edgecolor='black', facecolor='black', zorder=-2)
ax.scatter(0, 0, edgecolor='black', facecolor='white', s=20, clip_on=False, zorder=3)
ax.axhline(0, color='grey', linewidth=0.5, linestyle="dashed", zorder=0)
ax.axvline(1, color='grey', linewidth=0.5, linestyle="dashed", zorder=0)

ax.set_title("g_circ_l generator")

ax = plt.subplot(1, 3, 2)
c_grid_0, v_grid_0 = d_dL.grid
c_grid_1, v_grid_1 = d_dR.grid
ax.quiver(*c_grid_0, *v_grid_0, scale=20, color=spot_color)
ax.quiver(*c_grid_1, *v_grid_1, scale=20, color='black')
ax.set_aspect('equal')
ax.set_xlim(0, 2.25)
ax.set_ylim(-1.1, 1.25)
# ax.scatter(1, 0, edgecolor='black', facecolor='black', zorder=-2)
ax.scatter(0, 0, edgecolor='black', facecolor='white', s=20, clip_on=False, zorder=3)
ax.axhline(0, color='grey', linewidth=0.5, linestyle="dashed", zorder=0)
ax.axvline(1, color='grey', linewidth=0.5, linestyle="dashed", zorder=0)
circle = plt.Circle(g.value, .2, edgecolor='black',
                    facecolor='none', linestyle='dashed')
ax.add_artist(circle)




ax = plt.subplot(1, 3, 3)
c_grid_1, v_grid_1 = d_dR.grid
ax.quiver(*c_grid_1, *v_grid_1, scale=20, color='black')
ax.set_aspect('equal')
ax.set_xlim(0, 2.25)
ax.set_ylim(-1.1, 1.25)
# ax.scatter(1, 0, edgecolor='black', facecolor='black', zorder=-2)
ax.scatter(0, 0, edgecolor='black', facecolor='white', s=20, clip_on=False, zorder=3)
ax.axhline(0, color='grey', linewidth=0.5, linestyle="dashed", zorder=0)
ax.axvline(1, color='grey', linewidth=0.5, linestyle="dashed", zorder=0)

ax.set_title("g_circ_right generator")

plt.show()
