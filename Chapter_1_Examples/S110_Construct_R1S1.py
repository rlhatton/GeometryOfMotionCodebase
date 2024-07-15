#! /usr/bin/python3
import numpy as np
from geomotion import manifold as md, utilityfunctions as ut


# Cylinder with two charts positioned so that their first axes are aligned with the
# circumference, on opposite sides of each other, and with unit circumference as
# measured in the chart

# Define tranition maps between the two charts
def front_to_back(front_coords):
    back_coords = np.empty_like(front_coords)
    back_coords[0] = ut.cmod(front_coords[0] + 0.5, 1)
    back_coords[1] = front_coords[1]

    return back_coords


def back_to_front(back_coords):
    front_coords = np.empty_like(back_coords)
    front_coords[0] = ut.cmod(back_coords[0] - 0.5, 1)
    front_coords[1] = back_coords[1]

    return front_coords


# Construct transition table
transition_table = [[None, front_to_back], [back_to_front, None]]

# Generate the manifold
R1S1 = md.Manifold(transition_table, 2)
