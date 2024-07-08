import numpy as np
from geomotion import manifold as md, diffmanifold as tb, utilityfunctions as ut, plottingfunctions as gplt
from S400_Construct_R2 import R2
from matplotlib import pyplot as plt

# Make things print nicely
np.set_printoptions(precision=2)
spot_color = gplt.crimson


# Define a polar displacement in Cartesian coordinates
def f_rho_phi(xy, rp):
    x = xy[0]
    y = xy[1]
    rho = rp[0]
    phi = rp[1]

    rotmatrix = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])

    new_xy = (1 + rho / (np.sqrt(np.square(x) + np.square(y)))) * np.matmul(rotmatrix, np.array([[x], [y]]))

    return np.squeeze(new_xy)


# Define a polar scaling in Cartesian coordinates
def f_k_phi(xy, kp):
    x = xy[0]
    y = xy[1]
    k = kp[0]
    phi = kp[1]

    rotmatrix = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])

    print(rotmatrix)

    # Note that we need the 1+k so that the derivative is around scale=1
    new_xy = (1+k) * np.matmul(rotmatrix, np.array([[x], [y]]))

    return np.squeeze(new_xy)


# Define functions that apply equal radial and rotational motion under each function
def f_11rp(xy, delta):
    return f_rho_phi(xy, np.concatenate((delta, delta)))


def f_11kp(xy, delta):
    return f_k_phi(xy, np.concatenate((delta, delta)))


# Turn these functions into maps from R2 to itself
f_11rp_map = md.ManifoldMap(R2, R2, f_11rp)
f_11kp_map = md.ManifoldMap(R2, R2, f_11kp)

# Get the derivatives in the directions of these functions
f_11rp_dirdiv = tb.DirectionDerivative(f_11rp_map)
f_11kp_dirdiv = tb.DirectionDerivative(f_11kp_map)

f_11kp_dirdiv(R2.element([1, 0]))

# Construct a grid over which to evaluate the vector fields
grid_rt = ut.meshgrid_array([1, 2], np.linspace(-np.pi, np.pi, 9))
grid_points = R2.element_set(grid_rt, 1) #.transition(0)

# Evaluate the vector fields on the grids
f_11rp_vectors = f_11rp_dirdiv(grid_points)
f_11kp_vectors = f_11kp_dirdiv(grid_points)

ax = plt.subplot(1, 2, 1)
c_grid, v_grid = f_11rp_vectors.grid
ax.quiver(*c_grid, *v_grid, scale=15, color=spot_color)
ax.set_aspect('equal')
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
circle = plt.Circle([0, 0], 1, edgecolor='black',
                    facecolor='none', linestyle='dashed')
ax.add_artist(circle)
circle = plt.Circle([0, 0], 2, edgecolor='black',
                    facecolor='none', linestyle='dashed')
ax.add_artist(circle)
ax.set_title("Radial shift")

ax = plt.subplot(1, 2, 2)
c_grid, v_grid = f_11kp_vectors.grid
ax.quiver(*c_grid, *v_grid, scale=15, color=spot_color)
ax.set_aspect('equal')
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
circle = plt.Circle([0, 0], 1, edgecolor='black',
                    facecolor='none', linestyle='dashed')
ax.add_artist(circle)
circle = plt.Circle([0, 0], 2, edgecolor='black',
                    facecolor='none', linestyle='dashed')
ax.add_artist(circle)
ax.set_title("Radial scale")



plt.show()