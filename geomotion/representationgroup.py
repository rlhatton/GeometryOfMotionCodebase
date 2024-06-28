#! /usr/bin/python3
import numpy as np
from geomotion import utilityfunctions as ut
from geomotion import group as gp


class RepresentationGroup(gp.Group):

    def __init__(self,
                 representation_function_list,
                 identity,
                 derepresentation_function_list=None,
                 specification_chart=0,
                 ):

        # Regularize representation and derepresentation function lists, wrapping them in tuples if provided as raw
        # functions
        representation_function_list = ut.ensure_tuple(representation_function_list)
        derepresentation_function_list = ut.ensure_tuple(derepresentation_function_list)

        # If a derepresentation list has been provided, use it to construct the transition map as the composition of
        # the rep and derep functions
        if ((derepresentation_function_list is not None)
                and (len(derepresentation_function_list) == len(representation_function_list))):
            transition_table = [
                [lambda x: derepresentation_function_list[j](representation_function_list[i](x)) for j in range(2)] for
                i in range(len(representation_function_list))]
        else:
            transition_table = ((None,))

        # Make sure that we have both the representation of the identity (for constructing the group) and its
        # derepresentation (for determining the dimensionality)

        # Make sure that the identity is specified an ndarray
        identity = ut.ensure_ndarray(identity)

        # make sure that we have the identity in both the matrix and coordinate-list forms
        if identity.ndim == 2:
            identity_representation = identity
            identity_derepresentation = derepresentation_function_list[specification_chart](identity)
        else:
            identity_representation = representation_function_list[specification_chart](identity)
            identity_derepresentation = identity

        # Initialize the representation group as a group, using None for the attributes we are going to re-implement
        super().__init__(None,  # Operation list
                         identity_derepresentation,  # Identity list, for dimensionality calculation in Group class
                         None,  # Inverse function list
                         transition_table)

        # Save the representation function as an instance attribute,
        self.representation_function_list = representation_function_list

        # Save the derepresentation function as an instance attribute, wrapping it in a tuple if provided as a raw
        # function
        self.derepresentation_function_list = derepresentation_function_list

        # Store the identity input as the group identity representation
        self.identity_rep = identity_representation

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
            representation = group.identity_rep

        # Use the provided inputs to generate the group-element properties of the group element
        # Don't pass in an initial value; we are making value a property that depends on the representation and chart
        super().__init__(group,
                         group.identity_list[0],
                         initial_chart)

        # Save the representation (using the type enforcement in the setter)
        self.rep = representation

    def L(self,
          g_right):

        g_composed_rep = np.matmul(self.rep, g_right.rep)

        return RepresentationGroupElement(self.group,
                                          g_composed_rep,
                                          self.current_chart)

    def R(self,
          g_left):
        g_composed_rep = np.matmul(g_left.rep, self.rep)

        return RepresentationGroupElement(self.group,
                                          g_composed_rep,
                                          self.current_chart)

    @property
    def inverse(self):

        g_inv_rep = np.linalg.inv(self.rep)

        g_inv = self.group.element(g_inv_rep)

        return g_inv

    @property
    def rep(self):
        return self._representation

    @rep.setter
    def rep(self,
            representation):

        # Make sure that the provided representation is an ndarray
        representation = ut.ensure_ndarray(representation)

        # Force the representation into matrix form if it is not already in matrix form
        if representation.ndim == 2:
            pass
        else:
            representation = self.group.representation_function_list[self.current_chart](representation)

        # Store the matrix representation
        self._representation = representation

    @property
    def value(self):

        val_raw = self.group.derepresentation_function_list[self.current_chart](self.rep)

        # Make sure that the value is a list or ndarray
        val = ut.ensure_ndarray(val_raw)

        return val

    @value.setter
    def value(self, val):

        # Pass the value input into the representation setter (which will force it to matrix form)
        self.rep = val
