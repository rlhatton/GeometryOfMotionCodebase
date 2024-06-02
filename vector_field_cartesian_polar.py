import numpy as np
import manifold as md
import tangent_bundle as tb
import utilityfunctions as ut


#########
# Define manifold transition functions
def polar_to_cartesian(polar_coords):
    cartesian_coords = np.copy(polar_coords)
    cartesian_coords[0] = polar_coords[0] * np.cos(polar_coords[1])
    cartesian_coords[1] = polar_coords[0] * np.sin(polar_coords[1])

    return cartesian_coords


def cartesian_to_polar(cartesian_coords):
    polar_coords = np.copy(cartesian_coords)
    polar_coords[0] = np.sqrt((cartesian_coords[0] * cartesian_coords[0]) + (cartesian_coords[1] * cartesian_coords[1]))
    polar_coords[1] = np.arctan2(cartesian_coords[1], cartesian_coords[0])

    return polar_coords


# Build transition table from transition functions
transition_table = [[None, cartesian_to_polar], [polar_to_cartesian, None]]

# Build the manifold
Q = md.Manifold(transition_table, 2)

# Define a vector field function that points outward everywhere
def v_outward_xy(q):
    v = np.array([[q[0]], [q[1]]])
    return v

# Use the vector field function to construct a vector field
X_outward_xy = tb.TangentVectorField(v_outward_xy, Q)

# Build a grid over which to evaluate the vector field
grid = ut.meshgrid_array([-1, 1, 2], [-2, 2, 3])

# Evaluate the vector field on the grid
vgrid = X_outward_xy.grid_evaluate_vector_field(grid)
print("The Cartesian components of the outward field are the same as the underlying Cartesian coordinates: \n", vgrid, "\n")

# Transition the vector field into polar coordinates
X_outward_rt = X_outward_xy.transition(1)

# Construct a grid of polar coordinates
grid_rt = ut.meshgrid_array([.5, 1, 2], [0, np.pi / 2, np.pi])

# Evaluate the polar-coordinate-expressed field on the polar grid
vgrid_rt = X_outward_rt.grid_evaluate_vector_field(grid_rt)

print("The polar components of the outward field are all in the radial direction: \n", vgrid_rt)
