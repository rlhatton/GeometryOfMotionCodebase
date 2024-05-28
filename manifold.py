#! /usr/bin/python3
import copy

import numpy as np


class Manifold:
    """
    Class to hold manifold structure
    """

    def __init__(self,
                 transition_table):
        # Save the provided chart transition table as a class instance attribute
        self.transition_table = transition_table
        # Extract the number of charts implied by the transition table
        self.n_charts = len(transition_table)


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
        copied_element = copy.deepcopy(self)
        copied_element.value = new_value
        copied_element.current_chart = new_chart
        return copied_element
