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
                 n_dim):
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

    def element_set(self,
                    *args):
        q_set = ManifoldElementSet(self, *args)

        return q_set

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
                 defining_chart=0,
                 postprocess_function=None):

        self.manifold = manifold
        self.defining_function = defining_function
        self.defining_chart = defining_chart
        self.postprocess_function = postprocess_function

    def __call__(self, configuration_value, *args, **kwargs):

        # Convert the configuration value into a GridArray of numeric data
        configuration_grid, value_type = self.preprocess(configuration_value)

        # Evaluate
        function_grid = self.process(configuration_grid, *args, **kwargs)

        # Apply any post-process formatting
        function_value = self.postprocess(configuration_grid, function_grid, value_type)

        return function_value

    def preprocess(self, configuration_value):

        if (isinstance(configuration_value, ManifoldElement) or
                (isinstance(configuration_value, np.ndarray) and
                 configuration_value.shape == self.manifold.element_shape)):

            # Record the single-input for post-processing
            value_type = 'single'

            # Extract the value if the provided input is a ManifoldElement
            if isinstance(configuration_value, ManifoldElement):
                configuration_value = configuration_value.transition(self.defining_chart).value

            # Make the value an element-first grid array for processing
            configuration_value = ut.GridArray([configuration_value], 1)

        elif isinstance(configuration_value, ManifoldElementSet) or isinstance(configuration_value, ut.GridArray):

            # Record the multiple-input for post-processing
            value_type = 'multiple'

            # Extract the list of vectors to an element-first GridArray if the input is a ManifoldElementSet
            if isinstance(configuration_value, ManifoldElementSet):
                configuration_value = configuration_value.grid

            # Format the grid to be element-first
            configuration_value = ut.format_grid(configuration_value, self.manifold.element_shape, 'element')

        else:

            raise Exception(
                "Configuration_value input does not appear to be a ManifoldElement, ManifoldElementSet, or a numeric "
                "type of an appropriate format (ndarray for singleton values, or GridArray for multiple values")

        return configuration_value, value_type

    def process(self, configuration_grid, *args, **kwargs):

        def defining_function_with_inputs(config):
            return self.defining_function(config, *args, **kwargs)

        function_grid = configuration_grid.grid_eval(defining_function_with_inputs)

        return function_grid

    def postprocess(self, configuration_grid, function_grid, value_type):

        if self.postprocess_function is not None:
            if value_type == 'single':
                function_value = function_grid[0]  # Extract the single output from the grid array
                configuration_value = configuration_grid[0]
                return self.postprocess_function[0](configuration_value, function_value)
            elif value_type == 'multiple':
                return self.postprocess_function[1](configuration_grid, function_grid)
            else:
                raise Exception("Value_type should be 'single' or 'multiple'.")
        else:
            return function_grid

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
                              new_chart,
                              self.postprocess_function)

    def pullback(self, pullback_function, *args, **kwargs):

        return core.PullbackFunction(self, pullback_function, *args, **kwargs)


class ManifoldMap(ManifoldFunction):

    def __init__(self,
                 manifold: Manifold,
                 output_manifold: Manifold,
                 defining_function,
                 defining_chart=0,
                 output_defining_chart=0,
                 output_chart=None):

        if output_chart is None:
            output_chart = output_defining_chart

        def postprocess_function_single(q_input, q_output):
            return output_manifold.element(q_output, output_defining_chart).transition(output_chart)

        def postprocess_function_multiple(q_input, q_output):
            return output_manifold.element_set(q_output, output_defining_chart).transition(output_chart)

        postprocess_function = [postprocess_function_single, postprocess_function_multiple]

        super().__init__(manifold,
                         defining_function,
                         defining_chart,
                         postprocess_function)

        self.output_defining_chart = output_defining_chart
        self.output_chart = output_chart
        self.output_manifold = output_manifold

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
                              self.output_manifold,
                              new_defining_function,
                              new_chart,
                              self.output_defining_chart,
                              self.output_chart)

    def transition_output(self, new_output_chart):
        return self.__class__(self.manifold,
                              self.output_manifold,
                              self.defining_function,
                              self.defining_chart,
                              self.output_defining_chart,
                              new_output_chart)

    def pullback(self, pullback_function, *args, **kwargs):

        return core.PullbackFunction(self, pullback_function, *args, **kwargs)
