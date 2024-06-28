#! /usr/bin/python3
from collections import UserList
import numpy as np
from operator import methodcaller
from . import utilityfunctions as ut
from . import core


class Manifold:
    """
    Class to hold manifold structure
    """

    def __init__(self,
                 transition_table,
                 n_dim,
                 immersion_table=None):
        # Save the provided chart transition table as a class instance attribute
        self.transition_table = transition_table
        # Extract the number of charts implied by the transition table
        self.n_charts = len(transition_table)
        # Save the provided dimensionality as a class instance attribute
        self.n_dim = n_dim

    def element(self,
                value,
                initial_chart=0):
        """Instantiate a manifold element with a specified value"""
        q = ManifoldElement(self,
                            value,
                            initial_chart)
        return q

    @property
    def element_shape(self):
        return (self.n_dim,)


class ManifoldElement(core.GeomotionElement):
    """
    Class for manifold elements
    """

    def __init__(self,
                 manifold,
                 value,
                 initial_chart=0):

        # Save the provided manifold, configuration, and initial chart as class instance attributes
        self.manifold = manifold

        # Make sure that the value is a list or ndarray
        if not (isinstance(value, list) or isinstance(value, np.ndarray)):
            value = [value]

        # Make sure the value is a numpy float array
        self.current_chart = initial_chart
        self.value = value

    def format_value(self, val):

        val = ut.ensure_ndarray(val)

        # Check element shape
        if val.shape == self.manifold.element_shape:
            pass
        else:
            raise Exception("Provided value does not match element shape for manifold")

        return val

    def transition(self, new_chart):
        """
        Change the chart used to describe the manifold element, updating the numbers describing
        the configuration so that the actual point on the manifold stays the same
        """

        # Simple passthrough behavior if chart is not actually changing
        if new_chart == self.current_chart:

            new_value = self.value

        # Raise an exception if the transition from the current to new chart is not defined
        elif self.manifold.transition_table[self.current_chart][new_chart] is None:

            raise Exception(
                "The transition from " + str(self.current_chart) + " to " + str(new_chart) + " is undefined.")

        # If transition is non-trivial and defined, use the specified transition function
        else:

            new_value = self.manifold.transition_table[self.current_chart][new_chart](self.value)

        return self.__class__(self.manifold, new_value, new_chart)


class ManifoldElementSet(core.GeomotionSet):
    """ Argument list should either be a list of manifold elements or
    Manifold, GridArray, initial_chart, component-or-element """

    def __init__(self, *args):
        """ args is either a list of Elements or
        a Manifold
        a GridArray of values
        an initial chart
        (optional) component or element specification for grid"""

        n_args = len(args)

        # Check if the first argument is a list of Elements of the specified type, and if so, use it directly
        if isinstance(args[0], list):
            if ut.object_list_all_instance(ManifoldElement, args[0]):
                value = args[0]
                manifold = ut.object_list_extract_first_entry(args[0]).manifold
            else:
                raise Exception("List input to ManifoldElementSet should contain ManifoldElement objects",
                                "not ", type(ut.object_list_extract_first_entry(args[0])))

        # If the first argument is a manifold, process the inputs as if they were ManifoldElement inputs
        # with values provided in a GridArray
        elif isinstance(args[0], Manifold):
            manifold = args[0]
            if isinstance(args[1], ut.GridArray):

                # Extract the grid array from the argument list
                grid = args[1]

                # Check if the format of the grids has been specified
                if n_args > 3:
                    input_format = args[3]
                else:
                    input_format = None

                # Make sure that the grid is in element-outer format
                grid = ut.format_grid(grid, manifold.element_shape, 'element', input_format)

                # Convert element-outer grid to a list of ManifoldElements, including passing any initial chart to
                # the manifold element function
                if n_args > 2:
                    initial_chart = args[2]
                else:
                    initial_chart = 0

                def manifold_element_construction_function(manifold_element_value):
                    return manifold.element(manifold_element_value, initial_chart)

                value = ut.object_list_eval(manifold_element_construction_function, grid, grid.n_outer)

            else:
                raise Exception(
                    "If first input to ManifoldElementSet is a Manifold, second input should be a GridArray")

        else:
            raise Exception("First argument to ManifoldSet should be either a list of "
                            "Elements or a Manifold")

        super().__init__(value)
        self.manifold = manifold

    @property
    def element_shape(self):
        return (self.manifold.n_dim,)

    @property
    def grid(self):
        def extract_value(x):
            return x.value

        # Get an array of the manifold element values, and use nested_stack to make it an ndarray
        element_outer_grid = ut.nested_stack(ut.object_list_eval(extract_value,
                                                                 self.value))

        # Convert this array into a GridArray
        element_outer_grid_array = ut.GridArray(element_outer_grid, n_inner=1)

        component_outer_grid_array = element_outer_grid_array.everse

        return component_outer_grid_array

    def transition(self, new_chart):
        transition_method = methodcaller('transition', new_chart)

        new_set = ut.object_list_eval(transition_method,
                                      self.value)

        return self.__class__(new_set)


class ManifoldFunction:

    def __init__(self,
                 manifold,
                 defining_function,
                 defining_chart=0):
        self.manifold = manifold
        self.defining_function = defining_function
        self.defining_chart = defining_chart

    def __call__(self, configuration_value, *args, **kwargs):
        # If the input is a ManifoldElement, extract its value
        if isinstance(configuration_value, ManifoldElement):
            configuration_value = configuration_value.transition(self.defining_chart)
            configuration_value = configuration_value.value

        # Pass the configuration value, and any additional input values, into the defining function
        return self.defining_function(configuration_value, *args, **kwargs)

    def transition(self, new_chart):

        # Pull back the function by mapping the input from the new chart into the old chart where the function was
        # defined
        def new_defining_function(configuration_value, *args, **kwargs):
            old_configuration_value = self.manifold.transition_table[new_chart][self.defining_chart](
                configuration_value)

            return self.defining_function(old_configuration_value,
                                          *args,
                                          **kwargs)

        return self.__class__(self.manifold,
                              new_defining_function,
                              new_chart)

    def grid(self,
             configuration_grid: ut.GridArray,
             chart=None,
             output_format='grid',
             *args,
             **kwargs):

        # If chart is not specified, use tangent vector field's defining chart
        if chart is None:
            chart = self.defining_chart

        # Take in data formatted with the outer grid indices corresponding to the dimensionality of the data and the
        # inner grid indices corresponding to the location of those data points

        # Verify that the configuration grid is a one-dimensional GridArray and that the dimensionality matches that
        # of the manifold
        if not isinstance(configuration_grid, ut.GridArray):
            raise Exception("Expected configuration_grid to be of type GridArray.")

        if configuration_grid.n_outer != 1:
            raise Exception("Expected n_outer to be 1 for the GridArray provided as configuration_grid.")

        if configuration_grid.shape[0] != self.manifold.n_dim:
            raise Exception("Expected the first axis of the GridArray provided as configuration_grid to match the "
                            "dimensionality of the manifold.")

        # Convert the data grid so that the outer indices correspond the location of the data points and the inner
        # indices correspond to the dimensionality of the data
        configuration_at_points = configuration_grid.everse

        # Evaluate the defining function at each data location to get the function value at that location
        def f(x):
            f_at_x = self.__call__(x, *args, **kwargs)
            return f_at_x

        # Perform the evaluation
        function_at_points = configuration_at_points.grid_eval(f)

        function_grid = function_at_points.everse

        # Output format
        if output_format == 'grid':
            return function_grid
        else:
            raise Exception("Unknown output format for ManifoldFunction.grid")
