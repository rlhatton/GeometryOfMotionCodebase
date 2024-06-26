#! /usr/bin/python3
import numpy as np
from geomotion import manifold as md


# Cylinder with two charts positioned so that their first axes are aligned with the
# circumference, on opposite sides of each other, and with unit circumference as
# measured in the chart

# Define tranition maps between the two charts
def front_to_back(front_coords):
    back_coords = front_coords
    back_coords[0] = np.mod(front_coords[0] + 0.5, 1)

    return back_coords


def back_to_front(back_coords):
    front_coords = back_coords
    front_coords[0] = np.mod(back_coords[0] - 0.5)

    return front_coords


# Construct transition table
transition_table = [[None, front_to_back], [back_to_front, None]]

# Generate the manifold
R1S1 = md.Manifold(transition_table, 2)
