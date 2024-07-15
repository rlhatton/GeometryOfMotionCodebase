#! /usr/bin/python3
import warnings
import numpy as np


def passthrough(x):
    return x


def ensure_tuple(value):
    """ Function that wraps an input value in a tuple if it is not already a tuple"""
    if isinstance(value, tuple):
        value_tuple = value
    else:
        value_tuple = (value,)  # comma creates the tuple

    return value_tuple

def ensure_list(value):
    """ Function that wraps an input value in a tuple if it is not already a tuple"""
    if isinstance(value, list):
        value_list = value
    else:
        value_list = [value]  # comma creates the tuple

    return value_list

def ensure_ndarray(value):
    if not isinstance(value, np.ndarray):
        if not isinstance(value, list):
            value = [value]
    value = np.array(value, dtype=float)

    return value


def row(vec):
    return vec[None]


def column(vec):
    return vec[:, None]


def evert_array(arr, n_outer):
    """Take a multidimensional ndarray whose first n_outer dimensions correspond to a primary grid and whose remaining
    dimensions correspond to a secondary grid, and return an array in which the primary and secondary grids are
    swapped"""

    output_array = arr

    return np.moveaxis(output_array, list(range(n_outer)), list(range(-n_outer, 0)))


def array_eval(func, arr, n_outer=None, depth=0):
    if n_outer is None:
        n_outer = arr.n_outer

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


def array_eval_pairwise(func, arr1, arr2, n_outer, depth=0):
    # Verify that arrays are of the same length
    if arr1.shape[:n_outer - depth] == arr2.shape[:n_outer - depth]:
        pass
    else:
        raise Exception("Cannot make pairwise evaluation of arrays for arrays of different shape")

    # Get the length of the array at the current depth
    sh = arr1.shape[0]

    # If we're not at the deepest level of the outer grid, iterate over the level we're at, calling array_eval on the
    # next-deeper layer and creating an array of the results
    if (depth + 1) < n_outer:
        return np.array([array_eval_pairwise(func, arr1[i], arr2[i], n_outer, depth + 1) for i in range(sh)])
    # If we've reached the deepest level of the grid, evaluate the function for each point at this level and store
    # the results in an array
    else:
        return np.array([func(arr1[i], arr2[i]) for i in range(sh)])


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


def object_list_eval_two_outputs(f, object_list, n_outer=None, depth=0):
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
        output = [[], []]
        for i in range(sh):
            f_i = object_list_eval_two_outputs(f, object_list[i], n_outer, depth + 1)
            for j in [0, 1]:
                output[j].append(f_i[j])
        return tuple(output)

    # If we've reached the target level of the list, evaluate the specified method for each point at this level and
    # store the results in a list
    else:
        output = [[], []]
        for i in range(sh):
            f_i = f(object_list[i])
            for j in [0, 1]:
                output[j].append(f_i[j])
        return tuple(output)


def object_list_eval_pairwise(f, object_list_1, object_list_2, n_outer=None, depth=0):
    # Get the length of the array at the current depth
    sh = len(object_list_1)

    # If a target dept was supplied, check if we've reached it
    if n_outer is not None:
        reached_target_depth = (depth + 1) >= n_outer
    # If no target depth was supplied, stop drilling down once we find a non-list item
    else:
        reached_target_depth = not all([isinstance(object_list_1[i], list) for i in range(sh)])

    # If we're not yet drilled down to the contents, recurse further down
    if not reached_target_depth:
        return [object_list_eval_pairwise(f, object_list_1[i], object_list_2[i], n_outer, depth + 1) for i in range(sh)]
    # If we've reached the target level of the list, evaluate the specified method for each point at this level and
    # store the results in a list
    else:
        return [f(object_list_1[i], object_list_2[i]) for i in range(sh)]


def object_list_eval_threewise(f, object_list_1, object_list_2, object_list_3, n_outer=None, depth=0):
    # Get the length of the array at the current depth
    sh = len(object_list_1)

    # If a target dept was supplied, check if we've reached it
    if n_outer is not None:
        reached_target_depth = (depth + 1) >= n_outer
    # If no target depth was supplied, stop drilling down once we find a non-list item
    else:
        reached_target_depth = not all([isinstance(object_list_1[i], list) for i in range(sh)])

    # If we're not yet drilled down to the contents, recurse further down
    if not reached_target_depth:
        return [object_list_eval_threewise(f, object_list_1[i], object_list_2[i], object_list_3[i], n_outer, depth + 1) for i in
                range(sh)]
    # If we've reached the target level of the list, evaluate the specified method for each point at this level and
    # store the results in a list
    else:
        return [f(object_list_1[i], object_list_2[i], object_list_3[i]) for i in range(sh)]


def object_list_eval_fourwise(f, object_list_1, object_list_2, object_list_3, object_list_4, n_outer=None, depth=0):
    # Get the length of the array at the current depth
    sh = len(object_list_1)

    # If a target dept was supplied, check if we've reached it
    if n_outer is not None:
        reached_target_depth = (depth + 1) >= n_outer
    # If no target depth was supplied, stop drilling down once we find a non-list item
    else:
        reached_target_depth = not all([isinstance(object_list_1[i], list) for i in range(sh)])

    # If we're not yet drilled down to the contents, recurse further down
    if not reached_target_depth:
        return [object_list_eval_fourwise(f, object_list_1[i], object_list_2[i], object_list_4[i], object_list_3[i],
                                          n_outer, depth + 1) for i in range(sh)]
    # If we've reached the target level of the list, evaluate the specified method for each point at this level and
    # store the results in a list
    else:
        return [f(object_list_1[i], object_list_2[i], object_list_3[i], object_list_4[i]) for i in range(sh)]


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

def object_list_method_eval_with_arg(method_name, object_list_1, object2, n_outer=None, depth=0):
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
        return [
            object_list_method_eval_with_arg(method_name, object_list_1[i], object2, n_outer, depth + 1)
            for i
            in range(sh)]
    # If we've reached the target level of the list, evaluate the specified method for each point at this level and
    # store the results in a list
    else:
        return [getattr(object_list_1[i], method_name)(object2) for i in range(sh)]


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


def grid_format_test(grid, element_shape):
    # Test if GridArray format could be component-wise in the outer dimension and element-wise on the
    # inner dimensions
    c_outer_e_inner = (grid.n_outer == len(element_shape)) and (grid.shape[:grid.n_outer] == element_shape)

    # Test if GridArray format could be element-wise in the outer dimension and component-wise on the
    # inner dimensions
    e_outer_c_inner = (grid.n_inner == len(element_shape)) and (grid.shape[-1:-1:-grid.n_outer] == element_shape)

    return c_outer_e_inner, e_outer_c_inner


def format_grid(grid, element_shape, target_format, input_format=None):
    # Verify target format and input_format strings are legitimate options
    if target_format in ['component', 'element']:
        pass
    else:
        raise Exception("target_format was not provided as 'component' or 'element' ")

    if (input_format is None) or (input_format in ['component', 'element']):
        pass
    else:
        raise Exception("input_format was provided as something other than None, 'component' or 'element' ")

    # Test if GridArray format could be component-wise in the outer dimension and element-wise on the
    # inner dimensions
    c_outer_e_inner = (grid.n_outer == len(element_shape)) and (grid.shape[:grid.n_outer] == element_shape)

    # Test if GridArray format could be element-wise in the outer dimension and component-wise on the
    # inner dimensions
    e_outer_c_inner = (grid.n_inner == len(element_shape)) and (grid.shape[-grid.n_inner:] == element_shape)

    if c_outer_e_inner and e_outer_c_inner:
        detected_format = None

    if c_outer_e_inner and (not e_outer_c_inner):
        detected_format = 'component'

    if (not c_outer_e_inner) and e_outer_c_inner:
        detected_format = 'element'

    if (not c_outer_e_inner) and (not e_outer_c_inner):
        # Grid is not compatible with either component or element structure
        raise Exception("Grid does not appear to be a component-wise or element-wise grid compatible with "
                        "the provided element shape")

    # Compare detected and input formats
    if detected_format is not None:
        if input_format is not None:
            if detected_format == input_format:
                output_format = detected_format
            else:
                raise Exception("Detected format does not match input format")
        else:
            output_format = detected_format
    else:
        if input_format is not None:
            output_format = input_format
        else:
            raise Exception("Grid format is ambiguous and was not specified.")

    # Convert the grid if necessary
    if output_format == target_format:
        pass
    else:
        grid = grid.everse

    return grid


# Zero-centered modulus operation
def cmod(value, span):
    return (np.mod(value + (.5 * span), span)) - (0.5 * span)


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
