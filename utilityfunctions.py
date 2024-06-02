#! /usr/bin/python3
import numpy as np


def ensureTuple(value):
    """ Function that wraps an input value in a tuple if it is not already a tuple"""
    if isinstance(value, tuple):
        value_tuple = value
    else:
        value_tuple = (value,)  # comma creates the tuple

    return value_tuple


def evert_array(arr, n_outer):
    """Take a multi-dimensional ndarray whose first n_outer dimensions correspond to a primary grid and whose remaining
    dimensions correspond to a secondary grid, and return an array in which the primary and secondary grids are
    swapped"""

    return np.moveaxis(arr, list(range(n_outer)), list(range(-n_outer, 0)))


def array_eval(func, arr, n_grid_dim, depth=0):

    # Get the shape of the array at the current depth
    sh = arr.shape

    # If we're not at the deepest level of the grid, iterate over the level we're at, calling array_eval on the
    # next-deeper layer and creating an array of the results
    if (depth + 1) < n_grid_dim:
        return np.array([array_eval(func, arr[i], n_grid_dim, depth+1) for i in range(sh[0])])
    # If we've reached the deepest level of the grid, evaluate the function for each point at this level and store
    # the results in an array
    else:
        return np.array([func(arr[i]) for i in range(sh[0])])
