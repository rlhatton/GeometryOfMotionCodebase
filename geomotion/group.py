#! /usr/bin/python3
import numpy as np
from . import utilityfunctions as ut
from . import manifold as md
from . import core
from operator import methodcaller


class Group(md.Manifold):

    def __init__(self,
                 operation_list,
                 identity_list,
                 inverse_function_list=None,
                 transition_table=((None,),)):
        # Ensure that the operation list, identity list, and inverse function list are all actually lists
        operation_list = ut.ensure_list(operation_list)

        if not isinstance(identity_list, list) or not isinstance(identity_list[0], (list, np.ndarray)):
            identity_list = [identity_list]

        inverse_function_list = ut.ensure_list(inverse_function_list)

        # Extract the dimensionality from the identity element
        n_dim = np.size(identity_list[0])

        # Initialize the group as a manifold
        # super().__init__(transition_table,
        #                  n_dim)
        md.Manifold.__init__(self,
                             transition_table,
                             n_dim)

        # Save the operation list as an instance attribute
        self.operation_list = operation_list

        # Save the identity list as an instance attribute
        self.identity_list = identity_list

        # Save the inverse function list as an instance attribute
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
        md.ManifoldElement.__init__(self,
                                    group,
                                    value,
                                    initial_chart)

        # Information about how to build a set of these objects
        self.plural = GroupElementSet

    @property
    def L(self):
        return md.ManifoldMap(self.group,
                              self.group,
                              [lambda x: f(self.value, x) for f in self.group.operation_list],
                              list(range(len(self.group.operation_list))))

    @property
    def R(self):
        return md.ManifoldMap(self.group,
                              self.group,
                              [lambda x: f(x, self.value) for f in self.group.operation_list],
                              list(range(len(self.group.operation_list))))

    def AD(self, other):
        g_inv = self.inverse
        AD_g_other = self * other * g_inv
        return AD_g_other

    # noinspection SpellCheckingInspection
    def AD_inv(self, other):
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

        if isinstance(other, GroupElement):
            return self.L(other)
        else:
            return NotImplemented

    def __rmul__(self, other):

        if isinstance(other, GroupElement):
            return self.R(other)
        else:
            return NotImplemented


def commutator(g: GroupElement, h: GroupElement):
    return g * h * g.inverse * h.inverse


class GroupElementSet(md.ManifoldElementSet):

    def __init__(self,
                 manifold,
                 value=None,
                 initial_chart=0,
                 input_format=None):

        md.ManifoldElementSet.__init__(self,
                                       manifold,
                                       value,
                                       initial_chart,
                                       input_format)

        # Information about what this set should contain
        self.single = GroupElement

    def group_set_action(self, other, action_name):

        # Test if the counterparty is also a geomotion set
        if isinstance(other, core.GeomotionSet):

            if self.shape == other.shape:

                # Get the list of objects out of the set, and apply the named action method from the elements in
                # self to the elements in other
                new_set = ut.object_list_method_eval_pairwise(action_name, self.value, other.value)

            else:

                raise Exception("Cannot apply a set of GroupElements to a set of a different size")

        else:

            # If acting on a single-element counterparty, preload that single argument into the method, then
            # evaluate it for all elements in self
            action = methodcaller(action_name, other)
            new_set = ut.object_list_eval(action, self.value)

        # Identify what kind of set should be constructed from these objects
        plural_type = ut.object_list_extract_first_entry(new_set).plural
        return plural_type(new_set)

    def L(self, other):

        return self.group_set_action(other, 'L')

    def R(self, other):

        return self.group_set_action(other, 'R')

    def AD(self, other):

        return self.group_set_action(other, 'AD')

    # noinspection SpellCheckingInspection
    def AD_inv(self, other):

        return self.group_set_action(other, 'AD_inv')

    def commutator(self, other):

        return self.group_set_action(other, 'commutator')

    def __mul__(self, other):

        if isinstance(other, (GroupElement, GroupElementSet)):
            return self.group_set_action(other, '__mul__')
        else:
            return NotImplemented

    def __rmul__(self, other):

        if isinstance(other, (GroupElement, GroupElementSet)):
            return self.group_set_action(other, '__rmul__')
        else:
            return NotImplemented
