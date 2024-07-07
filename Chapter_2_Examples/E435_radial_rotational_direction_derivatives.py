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


# Define special cases for radial-only, rotation-only, and equal radial-rotational displacements
def f_rho(xy, rho):
    return f_rho_phi(xy, np.concatenate((rho, [0])))


def f_phi(xy, phi):
    return f_rho_phi(xy, np.concatenate(([0], phi)))


def f_11(xy, delta):
    return f_rho_phi(xy, np.concatenate((delta, delta)))


# Turn these functions into maps from R2 to itself
f_rho_map = md.ManifoldMap(R2, R2, f_rho)
f_phi_map = md.ManifoldMap(R2, R2, f_phi)
f_11_map = md.ManifoldMap(R2, R2, f_11)

# Get the derivatives in the directions of these functions
f_rho_dirdiv = tb.DirectionDerivative(f_rho_map)
f_phi_dirdiv = tb.DirectionDerivative(f_phi_map)
f_11_dirdiv = tb.DirectionDerivative(f_11_map)

# # Generate vector fields from these direction derivatives
# f_rho_field = tb.TangentVectorField(f_rho_dirdiv)
# f_phi_field = tb.TangentVectorField(f_phi_dirdiv)
# f_11_field = tb.TangentVectorField(f_11_dirdiv)

# Construct a grid over which to evaluate the vector fields
grid_xy = ut.meshgrid_array(np.linspace(-2, 2, 6), np.linspace(-2, 2, 5))
grid_points = R2.element_set(grid_xy)

# Evaluate the vector fields on the grids
f_rho_vectors = f_rho_dirdiv(grid_points)
f_phi_vectors = f_phi_dirdiv(grid_points)
f_11_vectors = f_11_dirdiv(grid_points)

ax = plt.subplot(1, 3, 1)
c_grid_rho, v_grid_rho = f_rho_vectors.grid
c_grid_phi, v_grid_phi = f_phi_vectors.grid
ax.quiver(*c_grid_rho, *v_grid_rho, scale=15, color=spot_color)
ax.quiver(*c_grid_phi, *v_grid_phi, scale=15, color='black')
ax.set_aspect('equal')
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_title("Derivatives in \n radial and rotational \n directions")

ax = plt.subplot(1, 3, 2)
c_grid_11, v_grid_11 = f_11_vectors.grid
ax.quiver(*c_grid_11, *v_grid_11, scale=15, color='black')
ax.set_aspect('equal')
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_title("Derivative in \n [1, 1] \n direction")

ax = plt.subplot(1, 3, 3)
v_grid_sum = v_grid_rho + v_grid_phi
ax.quiver(*c_grid_11, *v_grid_sum, scale=15, color='black')
ax.set_aspect('equal')
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_title("Sum of \n [1, 0] and [0, 1] \n derivatives")

plt.show()