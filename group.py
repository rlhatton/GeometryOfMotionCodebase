#! /usr/bin/python3
import numpy as np
import utilityfunctions as ut
import manifold as md


class Group(md.Manifold):

    def __init__(self,
                 operation_list,
                 identity_list,
                 transition_table=[[None]]):

        # Initialize the group as a manifold
        super().__init__(transition_table)

        # Save the operation as an instance attribute, wrapping it in a tuple if provided as a raw function
        self.operation_list = ut.ensureTuple(operation_list)

        # Save the identity as an instance attribute, wrapping it in a tuple if provided as a raw value
        self.identity_list = ut.ensureTuple(identity_list)

    def element(self,
                value,
                initial_chart=0):

        """Instantiate a group element with a specified value"""
        g = GroupElement(self,
                         value,
                         initial_chart)
        return g

    def identity_element(self,
                         initial_chart=0):

        """Instantiate a group element at the identity"""
        g = GroupElement(self,
                         'identity',
                         initial_chart)

        return g


class GroupElement(md.ManifoldElement):

    def __init__(self,
                 group,
                 value,
                 initial_chart=0):

        # Handle the identity-element value keyword
        if (value is 'identity'):
            if (group.identity_list[initial_chart] is not None):
                value = group.identity_list[initial_chart]
            else:
                raise Exception("The specified chart " + str(initial_chart) + "does not have an identity element "
                                                                              "specified")

        # Use the provided inputs to generate the manifold-element properties of the group element
        super().__init__(group,
                         value,
                         initial_chart)

        # Store the group into the group element attributes
        self.group = group

    def left_action(self,
                    g_right):

        # Attempt to ensure that g_right is expressed in the same chart as this group element
        g_right = g_right.transition(self.current_chart)

        # Apply the operation for the current chart, with this element on the left
        g_composed_value = self.group.operation_list[self.current_chart](self.value, g_right.value)

        # Construct an element from the composed value, in this element's chart
        g_composed = GroupElement(self.group, g_composed_value, self.current_chart)

        return g_composed

    def right_action(self,
                     g_left):

        # Attempt to ensure that g_left is expressed in the same chart as this group element
        g_left = g_left.transition(self.current_chart)

        # Apply the operation for the current chart, with this element on the right
        g_composed_value = self.group.operation_list[self.current_chart](g_left.value, self.value)

        # Construct an element from the composed value, in this element's chart
        g_composed = GroupElement(self.group, g_composed_value, self.current_chart)

        return g_composed

    def __mul__(self, other):

        return self.left_action(other)

    def __rmul__(self, other):

        return self.right_action(other)








