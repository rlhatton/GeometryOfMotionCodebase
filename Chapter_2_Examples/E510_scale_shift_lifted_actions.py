import numpy as np
from geomotion import manifold as md, diffmanifold as tb, utilityfunctions as ut, plottingfunctions as gplt
from S500_Construct_RxRplus import RxRplus
from matplotlib import pyplot as plt

# Make things print nicely
np.set_printoptions(precision=2)
spot_color = gplt.crimson

# Make the working group the scale-shift group
G = RxRplus


g_0 = G.identity_element()
g_1 = G.element([2, 3])

v_0 = G.vector(g_0, [1, 1])

v_1 = g_1.L_lifted(v_0)

print(v_1)