#! /usr/bin/python3
import numpy as np
from geomotion import manifold as md


def polar_to_cartesian(polar_coords):
    cartesian_coords = np.empty_like(polar_coords)
    cartesian_coords[0] = polar_coords[0] * np.cos(polar_coords[1])
    cartesian_coords[1] = polar_coords[0] * np.sin(polar_coords[1])

    return cartesian_coords


def cartesian_to_polar(cartesian_coords):
    polar_coords = np.empty_like(cartesian_coords)
    polar_coords[0] = np.sqrt((cartesian_coords[0] * cartesian_coords[0]) + (cartesian_coords[1] * cartesian_coords[1]))
    polar_coords[1] = np.arctan2(cartesian_coords[1], cartesian_coords[0])

    return polar_coords


transition_table = [[None, cartesian_to_polar], [polar_to_cartesian, None]]

R2 = md.Manifold(transition_table, 2)