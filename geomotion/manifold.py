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
                    value=None,
                    initial_chart=0,
                    input_format=None):

        q_set = ManifoldElementSet(self,
                                   value,
                                   initial_chart,
                                   input_format)

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
                 value=None,
                 initial_chart=0):

        # Check if the first argument is a ManifoldElementSet already, and if so, extract its value and manifold
        if isinstance(manifold, ManifoldElementSet):
            manifold_element_input = manifold
            self.value = manifold_element_input.value
            self.manifold = manifold_element_input.manifold
            self.current_chart = manifold_element_input.current_chart

        else:
            # Save the provided manifold, configuration, and initial chart as class instance attributes
            self.manifold = manifold

            # Make sure that the value is an ndarray
            value = ut.ensure_ndarray(value)

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

    def __init__(self,
                 manifold,
                 value=None,
                 initial_chart=0,
                 input_format=None):
        """ First input is one of:
        1. A ManifoldElementSet (which gets passed through into a copy of the original)
        2. A ManifoldElement (which gets wrapped into a single-element list)
        3. A nested list-of-lists of ManifoldElements
        4. a Manifold, followed by
           a GridArray of values
           an initial chart
           (optional) component-outer or element-outer specification for grid"""


        # Check if the first argument is a ManifoldElementSet already, and if so, extract its value and manifold
        if isinstance(manifold, ManifoldElementSet):
            manifold_element_set_input = manifold
            value = manifold_element_set_input.value
            manifold = manifold_element_set_input.manifold

        # Check if the first argument is a bare manifold element, and if so, wrap its value in a list
        elif isinstance(manifold, ManifoldElement):
            manifold_element_input = manifold
            value = [manifold_element_input]
            manifold = manifold_element_input.manifold

        # Check if the first argument is a list-of-lists of Elements of the specified type, and if so, use it directly
        elif isinstance(manifold, list):
            list_input = manifold
            if ut.object_list_all_instance(ManifoldElement, list_input):
                value = list_input
                manifold = ut.object_list_extract_first_entry(list_input).manifold
            else:
                raise Exception("List input to ManifoldElementSet should contain ManifoldElement objects",
                                "not ", type(ut.object_list_extract_first_entry(list_input)))

        # If the first argument is a manifold, process the inputs as if they were ManifoldElement inputs
        # with values provided in a GridArray
        elif isinstance(manifold, Manifold):
            if isinstance(value, ut.GridArray):

                # Extract the grid array from the argument list
                grid = value

                # Make sure that the grid is in element-outer format
                grid = ut.format_grid(grid, manifold.element_shape, 'element', input_format)

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

    def __call__(self, configuration, *args, **kwargs):

        # Convert the configuration value into an element-wise GridArray of numeric data
        configuration_grid_e, value_type = self.preprocess(configuration)

        # Evaluate the defining function over the grid with any additional arguments provided
        function_grid_e = self.process(configuration_grid_e, *args, **kwargs)

        # Apply any post-process formatting
        function_value = self.postprocess(configuration_grid_e, function_grid_e, value_type)

        return function_value

    def preprocess(self, configuration):
        """Configuration_value input can be one of:
                    1. ManifoldElement
                    3. ManifoldElementSet"""

        if isinstance(configuration, ManifoldElement):
            # Record the single-input status for post-processing
            value_type = 'single'
        elif isinstance(configuration, ManifoldElementSet):
            value_type = 'multiple'
        else:
            raise Exception("ManifoldFunction must be called with a ManifoldElement or ManifoldElementSet")

        # Put the input into ManifoldElementSet form (no change is made if it is already a set)
        configuration_set = ManifoldElementSet(configuration)

        # Transition the ManifoldElementSet into the defining chart
        configuration_set_defining_chart = configuration_set.transition(self.defining_chart)

        # Extract a component-wise grid from the ManifoldElementSet and evert it to element-wise
        configuration_grid_e = configuration_set_defining_chart.grid.everse

        return configuration_grid_e, value_type

    def process(self, configuration_grid_e, *process_args, **kwargs):

        def defining_function_with_inputs(config):
            return self.defining_function(config, *process_args, **kwargs)

        # Evaluate the defining function over the grid
        function_grid_e = configuration_grid_e.grid_eval(defining_function_with_inputs)

        return function_grid_e

    def postprocess(self, configuration_grid_e, function_grid_e, value_type):

        if self.postprocess_function is not None:
            if value_type == 'single':
                configuration_value = configuration_grid_e[0] # Extract the single item from the config grid array
                function_value = function_grid_e[0]  # Extract the single item from the function grid array
                return self.postprocess_function[0](configuration_value, function_value)
            elif value_type == 'multiple':
                return self.postprocess_function[1](configuration_grid_e, function_grid_e)
            else:
                raise Exception("Value_type should be 'single' or 'multiple'.")
        else:
            # Default is to return a component-wise grid of the function value
            return function_grid_e.everse

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

    def transition_output(self, new_output_chart):
        return self.__class__(self.manifold,
                              self.output_manifold,
                              self.defining_function,
                              self.defining_chart,
                              self.output_defining_chart,
                              new_output_chart)

    # def pullback(self, pullback_function, *args, **kwargs):
    #     return core.PullbackFunction(self, pullback_function, *args, **kwargs)
