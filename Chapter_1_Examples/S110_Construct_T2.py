#! /usr/bin/python3
import numpy as np
from geomotion import manifold as md


# Torus. Charts are chosen such that in a "doughnut" embedding:
# First chart's first axis is along the major circumference and second axis is along the minor circumference,
#
# Second chart also has first axis along major circumference, and is centered a quarter revolution away from the center
#   of the first chart along the minor axis
#
# Third chart is centered opposite both of the first two charts, aligned with first axis on the major axis

# Define tranition maps between the three charts
def first_to_second(input_coords):
    output_coords = input_coords
    output_coords[1] = np.mod(input_coords[1] - 0.25, 1)

    return output_coords


def first_to_third(input_coords):
    output_coords = input_coords
    output_coords[0] = np.mod(input_coords[0] - 0.5, 1)
    output_coords[1] = np.mod(input_coords[1] + 0.25, 1)

    return output_coords


def second_to_first(input_coords):
    output_coords = input_coords
    output_coords[1] = np.mod(input_coords[1] + 0.25, 1)

    return output_coords


def second_to_third(input_coords):
    output_coords = input_coords
    output_coords[0] = np.mod(input_coords[0] - 0.5, 1)
    output_coords[1] = np.mod(input_coords[1] + 0.5, 1)

    return output_coords


def third_to_first(input_coords):
    output_coords = input_coords
    output_coords[0] = np.mod(input_coords[0] + 0.5, 1)
    output_coords[1] = np.mod(input_coords[1] - 0.25, 1)

    return output_coords


def third_to_second(input_coords):
    output_coords = input_coords
    output_coords[0] = np.mod(input_coords[0] + 0.5, 1)
    output_coords[1] = np.mod(input_coords[1] - 0.5, 1)

    return output_coords


# Construct transition table
transition_table = [[None, first_to_second, first_to_third],
                    [second_to_first, None, second_to_third],
                    [third_to_first, third_to_second, None]]

# Generate the manifold
T2 = md.Manifold(transition_table, 2)
