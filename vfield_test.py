import numpy as np
from geomotion import utilityfunctions as ut


def vfun(p):
    print("input to vfun is " + str(p))
    return np.array([[p[0]], [p[1]]])


configuration_grid = ut.GridArray([[[-10, 20, 30], [-3, 4, 5]], [[10, 20, -30], [-3, -4, -5]]], 2)
print(configuration_grid.shape)
configuration_at_points = configuration_grid.everse
vectors_at_points = configuration_at_points.grid_eval(vfun)
vector_grid = vectors_at_points.everse

print(configuration_grid, '\n', configuration_at_points, '\n', vectors_at_points, '\n', vector_grid)
