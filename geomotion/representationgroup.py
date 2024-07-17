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
                 normalization_function=None,
                 ):

        # Regularize representation function list, wrapping it in list if provided as raw
        # function
        representation_function_list = ut.ensure_list(representation_function_list)
        representation_function_list = [lambda x: ut.ensure_ndarray(rho(x)) for rho in representation_function_list]

        # If a derepresentation list has been provided, use it to construct the transition map as the composition of
        # the rep and derep functions
        if derepresentation_function_list is not None:

            derepresentation_function_list = ut.ensure_list(derepresentation_function_list)

            if len(derepresentation_function_list) == len(representation_function_list):

                derepresentation_function_list = ut.ensure_list(derepresentation_function_list)

                derepresentation_function_list = [lambda x: ut.ensure_ndarray(rho(x)) for rho in derepresentation_function_list]

                transition_table = [
                    [lambda x: derepresentation_function_list[j](derepresentation_function_list[i](x)) for j in range(2)] for
                    i in range(len(derepresentation_function_list))]
            else:
                raise Exception("derepresentation function list does not have the same dimension as representation "
                                "function list")
        else:
            derepresentation_function_list = ut.ensure_list(derepresentation_function_list)
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

        # Save the normalization function list as an instance attribute
        self.normalization_function = normalization_function

        # Store the identity input as the group identity representation
        self.identity_rep = identity_representation
        self.identity_derep = identity_derepresentation

    def element(self,
                representation,
                initial_chart=0):

        """Instantiate a group element with a specified value"""
        g = RepresentationGroupElement(self,
                                       representation,
                                       initial_chart)
        return g

    def element_set(self,
                    value=None,
                    initial_chart=0,
                    input_format=None):

        g_set = RepresentationGroupElementSet(self,
                                              value,
                                              initial_chart,
                                              input_format)

        return g_set

    def identity_element(self,
                         initial_chart=0):

        """Instantiate a group element at the identity"""
        g = RepresentationGroupElement(self,
                                       'identity',
                                       initial_chart)

        return g

    @property
    def representation_shape(self):
        return self.identity_rep.shape


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
        gp.GroupElement.__init__(self,
                                 group,
                                 group.identity_list[0],
                                 initial_chart)

        # Save the representation (using the type enforcement in the setter)
        self.rep = representation

        # # Information about how to build a set of these objects
        self.plural = RepresentationGroupElementSet


    def L(self, other):

        new_rep = np.matmul(self.rep, other.rep)

        if self.group.normalization_function is not None:
            new_rep = self.group.normalization_function(new_rep)

        new_element = self.group.element(new_rep, self.current_chart)

        return new_element

    def R(self, other):

        new_rep = np.matmul(other.rep, self.rep)

        if self.group.normalization_function is not None:
            new_rep = self.group.normalization_function(new_rep)

        new_element = self.group.element(new_rep, self.current_chart)

        return new_element


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
            representation = ut.ensure_ndarray(
                self.group.representation_function_list[self.current_chart](representation))

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


class RepresentationGroupElementSet(gp.GroupElementSet):
    """This is mostly a pass-through copy of group element set, but allows
    us to set the self.single attribute"""

    def __init__(self,
                 group,
                 representation=None,
                 initial_chart=0,
                 input_format=None):
        gp.GroupElementSet.__init__(self,
                                    group,
                                    representation,
                                    initial_chart,
                                    input_format)

        # Information about what this set should contain
        self.single = RepresentationGroupElement
