#! /usr/bin/python3
import copy
import warnings

import numpy as np
from operator import methodcaller
from . import utilityfunctions as ut


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


class ManifoldElement:
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
        self.value = np.array(value, dtype=float)
        self.current_chart = initial_chart

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

        # Return a ManifoldElement with the new value and chart
        # copied_element = copy.deepcopy(self)
        # copied_element.value = new_value
        # copied_element.current_chart = new_chart

        return self.__class__(self.manifold, new_value, new_chart)

    def __getitem__(self, item):
        return self.value[item]

    def __str__(self):
        return str(self.value)


class ManifoldElementSet(ut.GeomotionSet):
    """ Argument list should either be a list of manifold elements or
    Manifold, GridArray, initial_chart, component-or-element """

    def __init__(self, *args):
        super().__init__(Manifold, ManifoldElement, *args)

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
