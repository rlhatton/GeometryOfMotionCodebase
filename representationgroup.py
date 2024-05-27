#! /usr/bin/python3
import numpy as np
import utilityfunctions as ut
import group as gp


class RepresentationGroup(gp.Group):

    def __init__(self,
                 representation_function_list,
                 identity,
                 specification_chart = 0,
                 transition_table=((None,))
                 ):

        # Initialize the representation group as a group, using None for the attributes we are going to re-implement
        super().__init__(None,              # Operation list
                         None,              # Identity list
                         None,              # Inverse function list
                         transition_table)

        # Store the identity input as the group identity representation, passing it through the appropriate
        # representation function if necessary
        if identity.ndim == 2:
            self.identity_representation = identity
        else:
            self.identity_representation = representation_function_list[specification_chart](identity)

    def element(self,
                value,
                initial_chart=0):

        """Instantiate a group element with a specified value"""
        g = RepresentationGroupElement(self,
                         value,
                         initial_chart)
        return g

    def identity_element(self,
                         initial_chart=0):

        """Instantiate a group element at the identity"""
        g = RepresentationGroupElement(self,
                         'identity',
                         initial_chart)

        return g


class RepresentationGroupElement(gp.GroupElement):

    def __init__(self,
                 group,
                 representation,
                 initial_chart=0):

        # Handle the identity-element value keyword
        if isinstance(representation, str) and (representation == 'identity'):
            representation = group.identity_representation

        # Use the provided inputs to generate the group-element properties of the group element
        super().__init__(group,
                         None,           # Don't pass in an initial value; we are making value a property that depends on the representation and chart
                         initial_chart)

        # Store the representation
        self.rep = representation

    def left_action(self,
                    g_right):

        g_composed_rep = np.matmul(self.rep, g_right.rep)

        return g_composed_rep

    def right_action(self,
                    g_left):
        g_composed_rep = np.matmul(g_left.rep, self.rep)

        return g_composed_rep


