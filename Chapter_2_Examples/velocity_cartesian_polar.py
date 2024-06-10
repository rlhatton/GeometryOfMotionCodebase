#! /usr/bin/python3
import numpy as np
from geomotion import diffmanifold as dm


def polar_to_cartesian(polar_coords):
    cartesian_coords = np.copy(polar_coords)
    cartesian_coords[0] = polar_coords[0] * np.cos(polar_coords[1])
    cartesian_coords[1] = polar_coords[0] * np.sin(polar_coords[1])

    return cartesian_coords


def cartesian_to_polar(cartesian_coords):
    polar_coords = np.copy(cartesian_coords)
    polar_coords[0] = np.sqrt((cartesian_coords[0]*cartesian_coords[0]) + (cartesian_coords[1]*cartesian_coords[1]))
    polar_coords[1] = np.arctan2(cartesian_coords[1], cartesian_coords[0])

    return polar_coords


transition_table = [[None, cartesian_to_polar], [polar_to_cartesian, None]]

Q = dm.DiffManifold(transition_table, 2)
q = Q.element([2, 0], 0)

v = Q.vector([[0], [1]], q)

v_polar = v.transition(1)

print("Polar expression of " + str(v.value) + " at " + str(q.value) + " is " + str(v_polar.value))

q = Q.element([1, 1], 0)

v = Q.vector([[1], [1]], q)

v_polar = v.transition(1)

print("Polar expression of " + str(v.value) + " at " + str(q.value) + " is " + str(v_polar.value))

v_polar_cartesian_chart = v.transition(1, 'keep')

print("Output vector has basis " + str(v_polar_cartesian_chart.current_basis) + " configuration chart " + str(v_polar_cartesian_chart.configuration.current_chart))
