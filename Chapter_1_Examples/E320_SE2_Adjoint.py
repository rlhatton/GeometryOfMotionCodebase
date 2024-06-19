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
h_A = SE2.element([-1, -1, -np.pi/2])
h_BwrtA = SE2.element([1, 0, np.pi/2])

# Adjoint-inverse action of the relative motion on the first frame's motion
ADi_hBA_hA = h_BwrtA.ADinv(h_A)

print("Given two frames A and B rigidly attached by a relative transformation h_B/A =", h_BwrtA, ","
      "\nthe local motion felt by B when A experiences a local motion h_A=", h_A,
      "\nis h_B = Adinv_h_B/A(h_A)=", ADi_hBA_hA)


a1 = RigidBody(cornered_triangle(.25, spot_color), e)
a2 = RigidBody(cornered_triangle(.25, spot_color), h_BwrtA)
a3 = RigidBody(cornered_triangle(.25, spot_color), h_A)
a4 = RigidBody(cornered_triangle(.25, spot_color), h_A * h_BwrtA)


ax = plt.subplot(1, 1, 1)
a1.draw(ax)
a2.draw(ax)
a3.draw(ax)
a4.draw(ax)
rect = plt.Rectangle([h_BwrtA[0]/2 - .75, h_BwrtA[1]/2-.35], 1.5, .7, edgecolor='black', facecolor='none', linestyle='dashed', zorder=-10)
ax.add_artist(rect)
rect = plt.Rectangle([((h_A*h_BwrtA)[0]+h_A[0])/2 - .35, ((h_A*h_BwrtA)[1]+h_A[1])/2-.75], .7, 1.5, edgecolor='black', facecolor='none', linestyle='dashed', zorder=-10)
ax.add_artist(rect)
ax.set_aspect('equal')
ax.arrow(e[0], e[1], .8 * (h_BwrtA[0] - e[0]), .8 * (h_BwrtA[1] - e[1]), zorder=0, length_includes_head=True, width=.01,
         head_width=0.1, overhang=.5, color='black')
ax.arrow(h_A[0], h_A[1], .8 * ((h_A*h_BwrtA)[0]-h_A[0]), .8 * ((h_A*h_BwrtA)[1]-h_A[1]), zorder=0, length_includes_head=True, width=.01,
         head_width=0.1, overhang=.5, color='black')
ax.arrow(e[0], e[1], .8 * (h_A[0] - e[0]), .8 * (h_A[1] - e[1]), zorder=0, length_includes_head=True, width=.01,
         head_width=0.1, overhang=.5, color=spot_color)
ax.arrow(h_BwrtA[0], h_BwrtA[1], .95 * ((h_A*h_BwrtA)[0]-h_BwrtA[0]), .95 * ((h_A*h_BwrtA)[1]-h_BwrtA[1]), zorder=0, length_includes_head=True, width=.01,
         head_width=0.1, overhang=.5, color=spot_color)


plt.show()

