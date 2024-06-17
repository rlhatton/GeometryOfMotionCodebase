from S300_Construct_SE2 import SE2
from S300_Construct_SE2 import RigidBody
from S300_Construct_SE2 import cornered_triangle
from geomotion import plottingfunctions as gplt
from matplotlib import pyplot as plt
import numpy as np

spot_color = gplt.crimson

e = SE2.identity_element()
g = SE2.element([2, 1, np.pi/2])
h = SE2.element([0, -1, -np.pi/4])

a = RigidBody(cornered_triangle(.25, spot_color), e)
b = RigidBody(cornered_triangle(.25, spot_color), g)
c = RigidBody(cornered_triangle(.25, spot_color), h)

ax = plt.subplot(1, 1, 1)
a.draw(ax)
b.draw(ax)
c.draw(ax)
ax.set_xlim(-1, 4)
ax.set_ylim(-2, 3)
ax.set_aspect('equal')
ax.axhline(0, color='black', zorder=-1)
ax.axvline(0, color='black', zorder=-1)


plt.show()
