#! /usr/bin/python3
import numpy as np


def ensure_tuple(value):
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


def array_eval(func, arr, n_outer, depth=0):
    # Get the length of the array at the current depth
    sh = arr.shape[0]

    # If we're not at the deepest level of the outer grid, iterate over the level we're at, calling array_eval on the
    # next-deeper layer and creating an array of the results
    if (depth + 1) < n_outer:
        return np.array([array_eval(func, arr[i], n_outer, depth + 1) for i in range(sh)])
    # If we've reached the deepest level of the grid, evaluate the function for each point at this level and store
    # the results in an array
    else:
        return np.array([func(arr[i]) for i in range(sh)])


def object_list_eval(method_function, object_list, target_depth=None, depth=0):
    # Get the length of the array at the current depth
    sh = len(object_list)

    # If a target dept was supplied, check if we've reached it
    if target_depth is not None:
        reached_target_depth = depth < target_depth
    # If no target depth was supplied, stop drilling down once we find a non-list item
    else:
        reached_target_depth = not all([isinstance(object_list[i], list) for i in range(sh)])


    # If we're not yet drilled down to the contents, recurse further down
    if not reached_target_depth:
        return [object_list_eval(method_function, object_list[i], target_depth, depth + 1) for i in range(sh)]
    # If we've reached the target level of the list, evaluate the specified method for each point at this level and
    # store the results in a list
    else:
        return [method_function(object_list[i]) for i in range(sh)]

def object_list_allinstance(test_class, object_list):
    def


def shape(a):
    if not isinstance(a, list):
        return []
    return [len(a)] + shape(a[0])


def meshgrid_array(*args):
    return GridArray(np.stack(np.meshgrid(*args)), 1)


class GridArray(np.ndarray):

    def __new__(cls, input_array, n_outer):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.n_outer = n_outer
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return
        self.info = getattr(obj, 'info', None)

    @property
    def everse(self):
        return GridArray(evert_array(self, self.n_outer), self.n_inner)

    @property
    def n_inner(self):
        return len(self.shape) - self.n_outer

    def grid_eval(self,
                  func):
        arr = array_eval(func, self, self.n_outer)
        garr = GridArray(arr, self.n_outer)
        return garr
