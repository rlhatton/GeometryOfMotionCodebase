from S300_Construct_SE2 import SE2
from S300_Construct_SE2 import RigidBody
from S300_Construct_SE2 import cornered_triangle
from geomotion import plottingfunctions as gplt
from matplotlib import pyplot as plt

spot_color = gplt.crimson

b = RigidBody(cornered_triangle(.25, spot_color))

ax = plt.subplot(1, 1, 1)
b.draw(ax)
ax.set_xlim(-1, 3)
ax.set_ylim(-1, 3)
ax.set_aspect('equal')


plt.show()
