import numpy as np
from geomotion import manifold as md, diffmanifold as tb, utilityfunctions as ut, plottingfunctions as gplt
from S500_Construct_RxRplus import RxRplus
from matplotlib import pyplot as plt

# Make things print nicely
np.set_printoptions(precision=2)
spot_color = gplt.crimson

G = RxRplus


def variable_group_composition_L(g_param, g_index, h_value):
    """ Function that creates a one-dimensional subgroup of the left group action along the g_index coordinate, from
    which the direction in the derivative of the left group action at h can be calculated"""

    # Get the group identity element
    g_value = G.identity_list[0].copy()

    # Modify the g_index value to be the provided g_param value (which numdifftools will supply as an array from which
    # the value needs to be extracted
    g_value[g_index] = g_value[g_index] + g_param[0]

    # Create group elements from the parameters
    g = G.element(g_value)
    h = G.element(h_value)

    # Compose the group elements
    gh = g * h

    return gh.value


def variable_group_composition_R(g_value, h_param, h_index):
    """ Function that creates a one-dimensional subgroup of the right group action along the h_index coordinate, from
    which the direction in the derivative of the right group action at g can be calculated"""

    # Get the group identity element
    h_value = G.identity_list[0].copy()

    # Modify the g_index value to be the provided h_param value (which numdifftools will supply as an array from which
    # the value needs to be extracted
    h_value[h_index] = h_value[h_index] + h_param[0]

    # Create group elements from the parameters
    h = G.element(h_value)
    g = G.element(g_value)

    # Compose the group elements
    gh = g * h

    return gh.value


# Create a grid in the right-half-plane
grid_xy = ut.meshgrid_array(np.linspace(.5, 2, 4), np.linspace(-1, 1, 5))
# Turn the grid into a set of points in the group
grid_points = RxRplus.element_set(grid_xy)

# Create a list of vector fields containing the derivatives in the directions
# of the left group actions in each parameter
d_dL = []
for i in range(G.n_dim):

    # Specialize the left group composition function to be along the ith direction at h
    def f_L(h, delta):
        return variable_group_composition_L(delta, i, h)

    # Equip the specialized composition function with the attributes of a ManifoldMap
    L_delta = md.ManifoldMap(G, G, f_L)

    # Take the directional derivative of the manifold map with respect to the delta variable, and save it to the list
    d_dL.append(tb.DirectionDerivative(L_delta)(grid_points))

# Create a list of vector fields containing the derivatives in the directions
# of the right group actions in each parameter
d_dR = []
for i in range(G.n_dim):

    # Specialize the right group composition function to be along the ith direction at h
    def f_R(g, delta):
        return variable_group_composition_R(g, delta, i)

    # Equip the specialized composition function with the attributes of a ManifoldMap
    R_delta = md.ManifoldMap(G, G, f_R)

    # Take the directional derivative of the manifold map with respect to the delta variable, and save it to the list
    d_dR.append(tb.DirectionDerivative(R_delta)(grid_points))


ax = plt.subplot(1, 2, 1)
c_grid_0, v_grid_0 = d_dL[0].grid
c_grid_1, v_grid_1 = d_dL[1].grid
ax.quiver(*c_grid_0, *v_grid_0, scale=20, color=spot_color)
ax.quiver(*c_grid_1, *v_grid_1, scale=20, color='black')
ax.set_aspect('equal')
ax.set_xlim(0, 2.25)
ax.set_ylim(-1.1, 1.25)
# ax.scatter(1, 0, edgecolor='black', facecolor='black', zorder=-2)
ax.scatter(0, 0, edgecolor='black', facecolor='white', s=20, clip_on=False, zorder=3)
ax.axhline(0, color='grey', linewidth=0.5, linestyle="dashed", zorder=0)
ax.axvline(1, color='grey', linewidth=0.5, linestyle="dashed", zorder=0)
ax.plot([0, c_grid_0[0][0][0]], [0, c_grid_0[1][0][0]], color='grey', linewidth=0.5, linestyle="dotted", zorder=0)
ax.plot([0, c_grid_0[0][1][2]], [0, c_grid_0[1][1][2]], color='grey', linewidth=0.5, linestyle="dotted", zorder=0)
ax.plot([0, c_grid_0[0][-1][-1]], [0, c_grid_0[1][-1][-1]], color='grey', linewidth=0.5, linestyle="dotted", zorder=0)
ax.plot([0, c_grid_0[0][-1][1]], [0, c_grid_0[1][-1][1]], color='grey', linewidth=0.5, linestyle="dotted", zorder=0)

ax.set_title("Left basis vectors")

ax = plt.subplot(1, 2, 2)
c_grid_0, v_grid_0 = d_dR[0].grid
c_grid_1, v_grid_1 = d_dR[1].grid
ax.quiver(*c_grid_0, *v_grid_0, scale=20, color=spot_color)
ax.quiver(*c_grid_1, *v_grid_1, scale=20, color='black')
ax.set_aspect('equal')
ax.set_xlim(0, 2.25)
ax.set_ylim(-1.1, 1.25)
# ax.scatter(1, 0, edgecolor='black', facecolor='black', zorder=-2)
ax.scatter(0, 0, edgecolor='black', facecolor='white', s=20, clip_on=False, zorder=3)
ax.axhline(0, color='grey', linewidth=0.5, linestyle="dashed", zorder=0)
ax.axvline(1, color='grey', linewidth=0.5, linestyle="dashed", zorder=0)

ax.set_title("Right basis vectors")

plt.show()
