#! /usr/bin/python3
from S275_Construct_RxRplus import RxRplus as RxRplus
from geomotion import utilityfunctions as ut
from geomotion import group as gp
from geomotion import plottingfunctions as gplt
from matplotlib import pyplot as plt
import numpy as np

G = RxRplus

# Set up the initial group elements
e = G.identity_element()
g1 = G.element([2, 2])
g2 = G.element([2, 3])
g3 = G.element([1, 2])

initial_points = gp.GroupElementSet([e, g1, g2, g3])

# squared_points = initial_points * initial_points

primed_points = g1.inverse * initial_points
print(primed_points[0], primed_points[1], primed_points[2], primed_points[3])

dprimed_points = initial_points * g1.inverse
print(dprimed_points[0], dprimed_points[1], dprimed_points[2], dprimed_points[3])


# initial_grid = ut.meshgrid_array([0, 2, 3, 4, 5, 6], [-2, -1, 0, 1, 2, 3, 4])
# transformed_grid =