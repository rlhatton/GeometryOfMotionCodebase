from S300_Construct_SE2 import SE2
from S300_Construct_SE2 import RigidBody
from S300_Construct_SE2 import cornered_triangle
from geomotion import plottingfunctions as gplt
from geomotion import group as gp
from geomotion import utilityfunctions as ut
from matplotlib import pyplot as plt
import numpy as np

np.set_printoptions(precision=2)  # Make things print nicely
spot_color = gplt.crimson

# Position of starting element
e = SE2.identity_element()

# Pure translation and pure rotation elements
g1 = SE2.element([2, 0, 0])
g2 = SE2.element([0, 0, np.pi / 3])

g1_g2 = g1 * g2
g2_g1 = g2 * g1

# Commutator of the elements
C_g1_g2 = gp.commutator(g1, g2)
g1_g2_g1inv = g1_g2 * g1.inverse

# Create lists of pure translation and pure rotation elements of different magnitudes
g1_list = [SE2.element([x, 0, 0]) for x in [1, 2, 3]]
g2_list = [SE2.element([0, 0, theta]) for theta in np.linspace(0, np.pi/2)]

# Convert the lists into GroupElementSets
g1_grid_elements = gp.GroupElementSet(g1_list)
g2_grid_elements = gp.GroupElementSet(g2_list)

# Calculate the commutator of each g1 element with each g2 element
C_grid_elements = gp.GroupElementSet(ut.object_list_method_eval_allpairs('commutator',
                                                                         g1_grid_elements.value,
                                                                         g2_grid_elements.value))

# Convert the commutator values back to a grid
C_grid = C_grid_elements.grid

print(C_grid.shape)

print("Moving an object at ", e, " by a right action of g1=", g1,
      "\nand then a right action of g2=", g2, " places it at g1*g2=", g1_g2, "."
                                                                             "\nMoving first by the right action of g2 and then the right action of g1"
                                                                             "\nInstead places it at g2*g1=", g2_g1, "."
                                                                                                                     "\n\n"
                                                                                                                     "The left action of the commutator [g1, g2]=",
      C_g1_g2, "bridges this difference,"
               "taking g2*g1 to g1*g2")

####
# Plotting stuff

# Make rigid bodies from the elements
a1 = RigidBody(cornered_triangle(.25, spot_color), e)
a2 = RigidBody(cornered_triangle(.25, 'grey', zorder=-1), g1)
a3 = RigidBody(cornered_triangle(.25, 'grey', zorder=-1), g2)
a4 = RigidBody(cornered_triangle(.25, spot_color), g1_g2)
a5 = RigidBody(cornered_triangle(.25, spot_color), g2_g1)
a6 = RigidBody(cornered_triangle(.25, 'grey', zorder=-1), g1_g2_g1inv)
a7 = RigidBody(cornered_triangle(.25, spot_color), C_g1_g2)

# Plotting commands
ax = plt.subplot(1, 2, 1)
a1.draw(ax)
a2.draw(ax)
a3.draw(ax)
a4.draw(ax)
a5.draw(ax)
circle = plt.Circle([e[0], e[1]], .3, edgecolor='black', facecolor='none', linestyle='dashed')
ax.add_artist(circle)
circle = plt.Circle([g1[0], g1[1]], .3, edgecolor='black', facecolor='none', linestyle='dashed', zorder=-2)
ax.add_artist(circle)
ax.arrow(e[0], e[1], .9 * (g1[0] - e[0]), .9 * (g1[1] - e[1]), zorder=0, length_includes_head=True, width=.01,
         head_width=0.2, overhang=.5, color='black')
ax.arrow(e[0], e[1], .9 * (g2_g1[0] - e[0]), .9 * (g2_g1[1] - e[1]), zorder=0, length_includes_head=True, width=.01,
         head_width=0.2, overhang=.5, color='black')
ax.arrow(g2_g1[0], g2_g1[1], .9 * (g1_g2[0] - g2_g1[0]), .9 * (g1_g2[1] - g2_g1[1]), zorder=0,
         length_includes_head=True, width=.01, head_width=0.2, overhang=.5, color=spot_color)
ax.set_aspect('equal')

ax = plt.subplot(1, 2, 2)
for i in range(C_grid.shape[0]):
    ax.plot(C_grid[0][i], C_grid[1][i], color='black', zorder=-1, linestyle='dotted')
a1.draw(ax)
a2.draw(ax)
a4.draw(ax)
a6.draw(ax)
a7.draw(ax)
circle = plt.Circle([g1[0], g1[1]], .3, edgecolor='black', facecolor='none', linestyle='dashed', zorder=-2)
ax.add_artist(circle)
circle = plt.Circle([C_g1_g2[0], C_g1_g2[1]], .3, edgecolor='black', facecolor='none', linestyle='dashed', zorder=-2)
ax.add_artist(circle)
ax.arrow(e[0], e[1], .9 * (g1[0] - e[0]), .9 * (g1[1] - e[1]), zorder=0, length_includes_head=True, width=.01,
         head_width=0.2, overhang=.5, color='black')
ax.arrow(g1[0], g1[1], .9 * (C_g1_g2[0] - g1[0]), .9 * (C_g1_g2[1] - g1[1]), zorder=0, length_includes_head=True, width=.01,
         head_width=0.2, overhang=.5, color='black')
ax.arrow(e[0], e[1], .9 * (C_g1_g2[0] - e[0]), .9 * (C_g1_g2[1] - e[1]), zorder=0, length_includes_head=True, width=.01,
         head_width=0.2, overhang=.5, color=spot_color)

ax.set_aspect('equal')


plt.show()
