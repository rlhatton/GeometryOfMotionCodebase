#! /usr/bin/python3
import numpy as np
import utilityfunctions as ut
import group as gp


class RepresentationGroup(gp.Group):

    def __init__(self,
                 representation_function_list,
                 identity,
                 derepresentation_function_list=None,
                 specification_chart=0,
                 transition_table=((None,))
                 ):

        # Initialize the representation group as a group, using None for the attributes we are going to re-implement
        super().__init__(None,  # Operation list
                         None,  # Identity list
                         None,  # Inverse function list
                         transition_table)

        # Save the representation function as an instance attribute, wrapping it in a tuple if provided as a raw
        # function
        self.representation_function_list = ut.ensureTuple(representation_function_list)

        # Save the derepresentation function as an instance attribute, wrapping it in a tuple if provided as a raw
        # function
        self.derepresentation_function_list = ut.ensureTuple(derepresentation_function_list)

        # Store the identity input as the group identity representation, passing it through the appropriate
        # representation function if necessary

        identity = np.array(identity, dtype=float)
        if identity.ndim == 2:
            identity_representation = identity
        else:
            identity_representation = self.representation_function_list[specification_chart](identity)

        self.identity_rep = np.array(identity_representation, dtype=float)

    def element(self,
                representation,
                initial_chart=0):

        """Instantiate a group element with a specified value"""
        g = RepresentationGroupElement(self,
                                       representation,
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
                         None, # Don't pass in an initial value; we are making value a property that depends on the
                               # representation and chart
                         initial_chart)

        # Store the representation, passing it through the representation function if necessary
        representation = np.array(representation, dtype=float)
        if representation.ndim == 2: #(np.squeeze(representation)).ndim == 2:
            pass
        else:
            representation = group.representation_function_list[initial_chart](representation)

        self.rep = np.array(representation, dtype=float)

    def left_action(self,
                    g_right):

        g_composed_rep = np.matmul(self.rep, g_right.rep)

        return RepresentationGroupElement(self.group,
                                          g_composed_rep,
                                          self.current_chart)
    def right_action(self,
                     g_left):
        g_composed_rep = np.matmul(g_left.rep, self.rep)

        return RepresentationGroupElement(self.group,
                                          g_composed_rep,
                                          self.current_chart)

    def inverse_element(self):

        g_inv_rep = np.linalg.inv(self.rep)

        g_inv = self.group.element(g_inv_rep)

        return g_inv


    @property
    def value(self):

        val = self.group.derepresentation_function_list[self.current_chart](self.rep)
        return val

    @value.setter
    def value(self, val):
        pass
