import numpy as np
from geomotion import manifold as md, diffmanifold as tb, utilityfunctions as ut, plottingfunctions as gplt
from S400_Construct_R2 import R2
from matplotlib import pyplot as plt

# Make things print nicely
np.set_printoptions(precision=2)
spot_color = gplt.crimson

# Make a single-chart 1-dimensional manifold
R1 = tb.DiffManifold([[None]], 1)


# Define the two function curves
def f1(q, delta):
    q_out = q + [delta[0], .1 * delta[0]] + [0, .5 * (np.cos(np.pi * delta[0] - np.pi / 12) - 1)] - [0,
                                                                                                     .5 * (np.cos(
                                                                                                         - np.pi / 12) - 1)]
    return q_out


def f2(q, delta):
    q_out = q + [-np.power(delta[0], 3), delta[0]]
    return q_out


# Define embedding maps for the two function curves, flipping the inputs so that the underlying points are deltas not qs
f1_embedding_map = md.ManifoldMap(R1, R2, lambda d, x: f1(x, d), 0, 0)
f2_embedding_map = md.ManifoldMap(R1, R2, lambda d, x: f2(x, d), 0, 0)

# Generate a set of points along a one-dimensional manifold
delta_points = R1.element_set(ut.GridArray([np.linspace(-1, 1)], n_outer=1), 0)

# Generate an "anchor point" for the curves to pass through
q0 = R2.element([0, 0])

# Embed the curves, passing q0 in via the *args in the ManifoldFunction processing structure
q1_points = f1_embedding_map(delta_points, q0.value)
q2_points = f2_embedding_map(delta_points, q0.value)

##########
# Generate the function derivative vectors by applying the differential embedding map to unit or twice-unit vectors
# along the one-dimensional manifolds
f1_diff_embedding_map = tb.DifferentialMap(f1_embedding_map)
f2_diff_embedding_map = tb.DifferentialMap(f2_embedding_map)

# Unit vector in the R1 space
delta_dot = R1.vector(0, 1)

# Immersion of the R1 unit vector based on each curve
delta_dot_f1 = f1_diff_embedding_map(delta_dot, q0.value)
delta_dot_f1_doubled = f1_diff_embedding_map(2 * delta_dot, q0.value)
delta_dot_f2 = f2_diff_embedding_map(delta_dot, q0.value)

##################
# Generate the same vectors as derivatives in the directions of the functions
f1_mapping = md.ManifoldMap(R2, R2, f1, 0, 0)
f2_mapping = md.ManifoldMap(R2, R2, f2, 0, 0)

f1_dirderiv = tb.DirectionDerivative(f1_mapping)
f2_dirderiv = tb.DirectionDerivative(f2_mapping)

d_df1 = f1_dirderiv(q0)
d_df1_doubled = 2 * d_df1
d_df2 = f2_dirderiv(q0)

ax = plt.subplot(2, 3, 1)
ax.scatter(*q0.value, color='black')
ax.plot(*q1_points.grid, color='black')
ax.quiver(*delta_dot_f1.configuration, *delta_dot_f1.value, scale=4, color=spot_color, width=.02)
ax.set_xlim(-1.25, 1.25)
ax.set_ylim(-1.2, .8)
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])

ax = plt.subplot(2, 3, 4)
ax.scatter(*q0.value, color='black')
ax.plot(*q1_points.grid, color='black')
ax.quiver(*d_df1.configuration, *d_df1.value, scale=4, color=spot_color, width=.02)
ax.set_xlim(-1.25, 1.25)
ax.set_ylim(-1.2, .8)
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])

ax = plt.subplot(2, 3, 2)
ax.scatter(*q0.value, color='black')
ax.plot(*q1_points.grid, color='black')
ax.quiver(*delta_dot_f1_doubled.configuration, *delta_dot_f1_doubled.value, scale=4, color=spot_color, width=.02)
ax.quiver(*delta_dot_f1.configuration, *delta_dot_f1.value, scale=4, color='grey', width=.02)
ax.set_xlim(-1.25, 1.25)
ax.set_ylim(-1.2, .8)
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
ax.set_title(r'Treat $f$ as immersion map and immerse $\dot{\delta}=1$ vector')

ax = plt.subplot(2, 3, 5)
ax.scatter(*q0.value, color='black')
ax.plot(*q1_points.grid, color='black')
ax.quiver(*d_df1_doubled.configuration, *d_df1_doubled.value, scale=4, color=spot_color, width=.02)
ax.quiver(*d_df1.configuration, *d_df1.value, scale=4, color='grey', width=.02)
ax.set_xlim(-1.25, 1.25)
ax.set_ylim(-1.2, .8)
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
ax.set_title(r"Treat $f$ as map from R2 to itself" + "\n" + "and calculate direction in derivative of this function")

ax = plt.subplot(2, 3, 3)
ax.scatter(*q0.value, color='black')
ax.plot(*q1_points.grid, color='black')
ax.plot(*q2_points.grid, color='black', linestyle='dashed')
ax.quiver(*delta_dot_f1.configuration, *delta_dot_f1.value, scale=4, color=spot_color, width=.02)
ax.quiver(*delta_dot_f2.configuration, *delta_dot_f2.value, scale=4, color=spot_color, width=.02)
ax.set_xlim(-1.25, 1.25)
ax.set_ylim(-1.2, .8)
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])

ax = plt.subplot(2, 3, 6)
ax.scatter(*q0.value, color='black')
ax.plot(*q1_points.grid, color='black')
ax.plot(*q2_points.grid, color='black', linestyle='dashed')
ax.quiver(*d_df1.configuration, *d_df1.value, scale=4, color=spot_color, width=.02)
ax.quiver(*d_df2.configuration, *d_df2.value, scale=4, color=spot_color, width=.02)
ax.set_xlim(-1.25, 1.25)
ax.set_ylim(-1.2, .8)
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])

plt.show()
