from S701_kinematic_chain import SE2, ground_point
from matplotlib import pyplot as plt
import numpy as np

e = SE2.identity_element()

gp = ground_point(e, .25)

gpw = ground_point(SE2.element([-3, 3, -np.pi/2]), .25)

ax = plt.subplot(1, 1, 1)
ax.set_aspect('equal')
gp.draw(ax)
gpw.draw(ax)

plt.show()