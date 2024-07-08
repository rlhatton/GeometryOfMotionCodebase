#! /usr/bin/python3
import numpy as np
from . import utilityfunctions as ut
from . import manifold as md
from operator import methodcaller


class Group(md.Manifold):

    def __init__(self,
                 operation_list,
                 identity_list,
                 inverse_function_list=None,
                 transition_table=((None,),)):
        # Ensure that the operation list, identity list, and inverse function list are all actually lists
        operation_list = ut.ensure_tuple(operation_list)
        identity_list = ut.ensure_tuple(identity_list)
        inverse_function_list = ut.ensure_tuple(inverse_function_list)

        # Extract the dimensionality from the identity element
        n_dim = np.size(identity_list[0])

        # Initialize the group as a manifold
        # super().__init__(transition_table,
        #                  n_dim)
        md.Manifold.__init__(self,
                             transition_table,
                             n_dim)

        # Save the operation list as an instance attribute, wrapping it in a tuple if provided as a raw function
        self.operation_list = operation_list

        # Save the identity list as an instance attribute, wrapping it in a tuple if provided as a raw value
        self.identity_list = identity_list

        # Save the inverse function list as an instance attribute, wrapping it in a tuple if provided as a raw function
        self.inverse_function_list = inverse_function_list

    def element(self,
                value,
                initial_chart=0):
        """Instantiate a group element with a specified value"""
        g = GroupElement(self,
                         value,
                         initial_chart)
        return g

    def element_set(self,
                    value=None,
                    initial_chart=0,
                    input_format=None):
        g_set = GroupElementSet(self,
                                value,
                                initial_chart,
                                input_format)

        return g_set

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
        if isinstance(value, str) and (value == 'identity'):
            if group.identity_list[initial_chart] is not None:
                value = group.identity_list[initial_chart]
            else:
                raise Exception("The specified chart " + str(initial_chart) + "does not have an identity element "
                                                                              "specified")

        # Use the provided inputs to generate the manifold-element properties of the group element
        super().__init__(group,
                         value,
                         initial_chart)

    def L(self,
          g_right):

        if self.group.operation_list[self.current_chart] is not None:

            # Attempt to ensure that g_right is expressed in the same chart as this group element
            g_right = g_right.transition(self.current_chart)

            # Apply the operation for the current chart, with this element on the left
            g_composed_value = self.group.operation_list[self.current_chart](self.value, g_right.value)

            # Construct an element from the composed value, in this element's chart
            g_composed = GroupElement(self.group, g_composed_value, self.current_chart)

        else:

            raise Exception("Group operation is undefined for chart " + str(self.current_chart))

        return g_composed

    def R(self,
          g_left):

        if self.group.operation_list[self.current_chart] is not None:

            # Attempt to ensure that g_left is expressed in the same chart as this group element
            g_left = g_left.transition(self.current_chart)

            # Apply the operation for the current chart, with this element on the right
            g_composed_value = self.group.operation_list[self.current_chart](g_left.value, self.value)

            # Construct an element from the composed value, in this element's chart
            g_composed = GroupElement(self.group, g_composed_value, self.current_chart)

        else:

            raise Exception("Group operation is undefined for chart " + str(self.current_chart))

        return g_composed

    def AD(self, other):
        g_inv = self.inverse
        AD_g_other = self * other * g_inv
        return AD_g_other

    # noinspection SpellCheckingInspection
    def ADinv(self, other):
        g_inv = self.inverse
        ADi_g_other = g_inv * other * self
        return ADi_g_other

    def commutator(self, other):
        return commutator(self, other)

    @property
    def inverse(self):

        g_inv_value = self.group.inverse_function_list[self.current_chart](self.value)

        g_inv = self.group.element(g_inv_value, self.current_chart)

        return g_inv

    @property
    def group(self):
        return self.manifold

    @group.setter
    def group(self, gp):
        self.manifold = gp

    def __mul__(self, other):

        if isinstance(other, GroupElement) and (self.manifold == other.manifold):
            return self.L(other)
        else:
            return NotImplemented

    def __rmul__(self, other):

        if isinstance(other, GroupElement) and (self.manifold == other.manifold):
            return self.R(other)
        else:
            return NotImplemented


def commutator(g: GroupElement, h: GroupElement):
    return g * h * g.inverse * h.inverse


class GroupElementSet(md.ManifoldElementSet):

    def group_set_action(self, other, action_name):

        if hasattr(other, 'shape'):

            if self.shape == other.shape:

                new_set = ut.object_list_method_eval_pairwise(action_name, self.value, other.value)

            else:

                raise Exception("Cannot apply a set of GroupElements to a set of a different size")

        else:

            action = methodcaller(action_name, other)
            new_set = ut.object_list_eval(action, self.value)

        return self.__class__(new_set)

    def L(self, other):

        return self.group_set_action(other, 'L')

    def R(self, other):

        return self.group_set_action(other, 'R')

    def AD(self, other):

        return self.group_set_action(other, 'AD')

    # noinspection SpellCheckingInspection
    def ADinv(self, other):

        return self.group_set_action(other, 'R')

    def commutator(self, other):

        return self.group_set_action(other, 'commutator')

    def __mul__(self, other):

        if (isinstance(other, GroupElement) or isinstance(other, GroupElementSet)) and (
                self.manifold == other.manifold):
            return self.group_set_action(other, '__mul__')
        else:
            return NotImplemented

    def __rmul__(self, other):

        if (isinstance(other, GroupElement) or isinstance(other, GroupElementSet)) and (
                self.manifold == other.manifold):
            return self.group_set_action(other, '__rmul__')
        else:
            return NotImplemented
