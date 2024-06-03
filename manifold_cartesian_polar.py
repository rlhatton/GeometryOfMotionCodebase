#! /usr/bin/python3
import numpy as np
import manifold as md


def polar_to_cartesian(polar_coords):
    cartesian_coords = np.copy(polar_coords)
    cartesian_coords[0] = polar_coords[0] * np.cos(polar_coords[1])
    cartesian_coords[1] = polar_coords[0] * np.sin(polar_coords[1])

    return cartesian_coords


def cartesian_to_polar(cartesian_coords):
    polar_coords = np.copy(cartesian_coords)
    polar_coords[0] = np.sqrt((cartesian_coords[0] * cartesian_coords[0]) + (cartesian_coords[1] * cartesian_coords[1]))
    polar_coords[1] = np.arctan2(cartesian_coords[1], cartesian_coords[0])

    return polar_coords


transition_table = [[None, cartesian_to_polar], [polar_to_cartesian, None]]

Q = md.Manifold(transition_table, 2)
q = Q.element([3, 3], 0)

print("Initial configuration in Cartesian coordinates is " + str(q.value))

q_polar = q.transition(1)

print("Configuration in polar coordinates " + str(q_polar.value))

q_cartesian = q_polar.transition(0)

print("Configuration back in Cartesian coordinates " + str(q_cartesian.value))
