#! /usr/bin/python3
import warnings
from collections import UserList
import numpy as np


class GeomotionSet(UserList):
    """ Generic class for sets of elements"""

    def __init__(self, ManifoldClass, contents_class, *args):

        n_args = len(args)

        # Check if the first argument is a list of Elements of the specified type, and if so, use it directly
        if isinstance(args[0], list):
            if object_list_all_instance(contents_class, args[0]):
                value = args[0]
                manifold = object_list_extract_first_entry(args[0]).manifold
            else:
                raise Exception("List input to GeomotionSet should contain Elements")

        # If the first argument is a manifold, process the inputs as if they were ManifoldElement inputs
        # provided in a GridArray
        elif isinstance(args[0], ManifoldClass):
            manifold = args[0]
            if isinstance(args[1], GridArray):

                # Extract the grid array from the argument list
                grid = args[1]

                # Test if GridArray format could be component-wise in the outer dimension and element-wise on the
                # inner dimensions
                c_outer_e_inner = (grid.n_outer == 1) and (grid.shape[0] == manifold.n_dim)

                # Test if GridArray format could be element-wise in the outer dimension and component-wise on the
                # inner dimensions
                e_outer_c_inner = (grid.n_inner == 1) and (grid.shape[-1] == manifold.n_dim)

                # Use results from these tests to determine grid format
                if c_outer_e_inner and e_outer_c_inner:

                    # Grid format is ambiguous. Check for fourth argument specifying which format is being used
                    if n_args > 3:
                        grid_format = args[3]
                        if grid_format == 'component':
                            grid = grid.everse
                        elif grid_format == 'element':
                            pass
                    else:
                        warnings.warn(
                            "Grid format is ambiguous and a grid format was not provided. Assuming component-outer grid")
                        grid = grid.everse

                if c_outer_e_inner and (not e_outer_c_inner):
                    # Convert component-outer grid to element-outer grid
                    grid = grid.everse
                    # print("Detected component-outer grid and everted it")

                if (not c_outer_e_inner) and e_outer_c_inner:
                    # Keep element-outer grid
                    pass
                    # print("Detected element-outer grid and maintained it")

                if (not c_outer_e_inner) and (not e_outer_c_inner):
                    # Grid is not compatible with manifold structure
                    raise Exception("Grid does not appear to be a component-wise or element-wise grid compatible with "
                                    "the provided manifold")

                # Convert element-outer grid to a list of ManifoldElements, including passing any initial chart to
                # the manifold element function
                if n_args > 2:
                    initial_chart = args[2]
                else:
                    initial_chart = 0

                def manifold_construction_function(x):
                    return manifold.element(x, initial_chart)

                value = object_list_eval(manifold_construction_function, grid, grid.n_outer)

            else:
                raise Exception(
                    "If first input to ManifoldElementSet is a Manifold, second input should be a GridArray")

        else:
            raise Exception("First argument to ManifoldElementSet should be either a list of "
                            "ManifoldElements or a Manifold")

        super().__init__(value)
        self.manifold = manifold

    @property
    def shape(self):
        return shape(self.value)

    @property
    def value(self):
        return self.data

    def __getitem__(self, item):
        return self.value[item]

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


def object_list_eval(f, object_list, n_outer=None, depth=0):
    # Get the length of the array at the current depth
    sh = len(object_list)

    # If a target dept was supplied, check if we've reached it
    if n_outer is not None:
        reached_target_depth = (depth + 1) >= n_outer
    # If no target depth was supplied, stop drilling down once we find a non-list item
    else:
        reached_target_depth = not all([isinstance(object_list[i], list) for i in range(sh)])

    # If we're not yet drilled down to the contents, recurse further down
    if not reached_target_depth:
        return [object_list_eval(f, object_list[i], n_outer, depth + 1) for i in range(sh)]
    # If we've reached the target level of the list, evaluate the specified method for each point at this level and
    # store the results in a list
    else:
        return [f(object_list[i]) for i in range(sh)]


def object_list_method_eval_pairwise(method_name, object_list_1, object_list_2, n_outer=None, depth=0):
    # Get the length of the first array at the current depth
    sh = len(object_list_1)

    # If a target dept was supplied, check if we've reached it
    if n_outer is not None:
        reached_target_depth = (depth + 1) >= n_outer
    # If no target depth was supplied, stop drilling down once we find a non-list item
    else:
        reached_target_depth = not all([isinstance(object_list_1[i], list) for i in range(sh)])

    # If we're not yet drilled down to the contents, recurse further down
    if not reached_target_depth:
        return [object_list_method_eval_pairwise(method_name, object_list_1[i], object_list_2[i], n_outer, depth + 1)
                for i
                in range(sh)]
    # If we've reached the target level of the list, evaluate the specified method for each point at this level and
    # store the results in a list
    else:
        return [getattr(object_list_1[i], method_name)(object_list_2[i]) for i in range(sh)]


def object_list_method_eval_allpairs(method_name, object_list_1, object_list_2, n_outer=None, depth=0):
    # Get the length of the first array at the current depth
    sh = len(object_list_1)

    # If a target dept was supplied, check if we've reached it
    if n_outer is not None:
        reached_target_depth = (depth + 1) >= n_outer
    # If no target depth was supplied, stop drilling down once we find a non-list item
    else:
        reached_target_depth = not all([isinstance(object_list_1[i], list) for i in range(sh)])

    # If we're not yet drilled down to the contents, recurse further down
    if not reached_target_depth:
        return [object_list_method_eval_allpairs(method_name, object_list_1[i], object_list_2, n_outer, depth + 1)
                for i
                in range(sh)]
    # If we've reached the target level of the list, evaluate the specified method for each point at this level and
    # store the results in a list
    else:
        return [object_list_eval(getattr(object_list_1[i], method_name), object_list_2) for i in range(sh)]


def object_list_all_instance(test_class, object_list):
    """ Check if all objects in a given nested list are instances of a given class"""

    # Function that errors out if the target is not of the right class
    def object_test(x):
        assert isinstance(x, test_class)

    try:
        object_list_eval(object_test, object_list)
        test_value = True
    except:
        test_value = False

    return test_value


def object_list_extract_first_entry(object_list, n_outer=None, depth=0):
    """Drill down through the list to extract the first item from it"""
    # Get the length of the array at the current depth
    sh = len(object_list)

    # If n_outer was supplied, check if we've reached it
    if n_outer is not None:
        reached_target_depth = (depth + 1) >= n_outer
    # If no target depth was supplied, stop drilling down once we find a non-list item
    else:
        reached_target_depth = not all([isinstance(object_list[i], list) for i in range(sh)])

    # If we're not yet drilled down to the contents, recurse further down
    if not reached_target_depth:
        return object_list_extract_first_entry(object_list[0])
    # If we've reached the target level of the list, evaluate the specified method for each point at this level and
    # store the results in a list
    else:
        return object_list[0]


def nested_stack(array_list, n_outer=None, depth=0):
    """Drill down through the list, stacking the arrays at each level"""
    # Get the length of the array at the current depth
    sh = len(array_list)

    # If n_outer was supplied, check if we've reached it
    if n_outer is not None:
        reached_target_depth = (depth + 1) >= n_outer
    # If no target depth was supplied, stop drilling down once we find a non-list item
    else:
        reached_target_depth = not all([isinstance(array_list[i], list) for i in range(sh)])

    # If we're not yet drilled down to the contents, recurse further down
    if not reached_target_depth:
        return np.stack([nested_stack(array_list[i]) for i in range(sh)])
    # If we've reached the target level of the list, evaluate the specified method for each point at this level and
    # store the results in a list
    else:
        return np.stack(array_list)


def shape(a):
    if not isinstance(a, list):
        return []
    return [len(a)] + shape(a[0])


def meshgrid_array(*args):
    return GridArray(np.stack(np.meshgrid(*args)), 1)


class GridArray(np.ndarray):

    def __new__(cls, input_array, n_outer=None, n_inner=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        if (n_outer is not None) and (n_inner is None):
            obj.n_outer = n_outer
        if (n_outer is None) and (n_inner is not None):
            obj.n_outer = len(obj.shape) - n_inner
        if (n_outer is not None) and (n_inner is not None):
            obj.n_outer = n_outer
            warnings.warn("Both n_outer and n_inner were specified. Using n_outer.")
        if (n_outer is None) and (n_inner is None):
            raise Exception("Must specify either n_outer or n_inner")
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


def format_radians_label(float_in):
    # Converts a float value in radians into a
    # string representation of that float
    string_out = str(float_in / (np.pi)) + "π"

    return string_out


def convert_polar_xticks_to_radians(ax):
    # Converts x-tick labels from degrees to radians

    # Get the x-tick positions (returns in radians)
    label_positions = ax.get_xticks()

    # Convert to a list since we want to change the type of the elements
    labels = list(label_positions)

    # Format each label (edit this function however you'd like)
    labels = [format_radians_label(label) for label in labels]

    ax.set_xticks(label_positions, labels)
