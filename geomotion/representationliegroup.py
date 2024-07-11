#! /usr/bin/python3
import numpy as np
from geomotion import utilityfunctions as ut
from geomotion import liegroup as lgp
from geomotion import diffmanifold as tb
from geomotion import representationgroup as rgp
import numdifftools as ndt


class RepresentationLieGroup(rgp.RepresentationGroup, lgp.LieGroup):

    def __init__(self,
                 representation_function_list,
                 identity,
                 derepresentation_function_list=None,
                 specification_chart=0,
                 ):
        # Instantiate as a representation group
        rgp.RepresentationGroup.__init__(self,
                                         representation_function_list,
                                         identity,
                                         derepresentation_function_list,
                                         specification_chart,
                                         )

        # Instantiate as a differentiable manifold
        # (Using this instead of LieGroup initialization to avoid initializing as group twice)
        tb.DiffManifold.__init__(self,
                                 self.transition_table,
                                 self.n_dim)

        # Construct the differential representation functions
        self.representation_Jacobian_table = \
            [ndt.Jacobian(rho) for rho in self.representation_function_list]



    def element(self,
                representation,
                initial_chart=0):
        """Instantiate a representation Lie group element with a specified value"""
        g = RepresentationLieGroupElement(self,
                                          representation,
                                          initial_chart)
        return g

    def element_set(self,
                    representation=None,
                    initial_chart=0,
                    input_format=None):
        g_set = RepresentationLieGroupElementSet(self,
                                                 representation,
                                                 initial_chart,
                                                 input_format)

        return g_set

    def vector(self,
               configuration,
               representation,
               initial_chart=0,
               initial_basis=0):
        """Instantiate a Lie grouptangent vector at a specified configuration on the manifold"""
        v = RepresentationLieGroupTangentVector(self,
                                                configuration,
                                                representation,
                                                initial_chart,
                                                initial_basis)
        return v

    def Lie_alg_vector(self,
                       representation,
                       initial_chart=0,
                       initial_basis=0):
        """Instantiate a Lie grouptangent vector at a specified configuration on the manifold"""
        v = RepresentationLieGroupTangentVector(self,
                                                self.identity_element(),
                                                representation,
                                                initial_chart,
                                                initial_basis)
        return v

    def vector_set(self,
                   configuration,
                   representation=None,
                   initial_chart=0,
                   initial_basis=0,
                   input_grid_format=None):
        v = RepresentationLieGroupTangentVectorSet(self,
                                                   configuration,
                                                   representation,
                                                   initial_chart,
                                                   initial_basis,
                                                   input_grid_format)
        return v


class RepresentationLieGroupElement(lgp.LieGroupElement, rgp.RepresentationGroupElement):

    def __init__(self,
                 group,
                 representation,
                 initial_chart=0):
        # Use RepresentationGroup for setup, because we're not using the inherited differential maps
        rgp.RepresentationGroupElement.__init__(self,
                                                group,
                                                representation,
                                                initial_chart)

        self.TL = lambda x: RepresentationLieGroupTangentVector(self.group,
                                                                self,
                                                                np.matmul(self.rep, x.rep),
                                                                self.current_chart)

        self.TR = lambda x: RepresentationLieGroupTangentVector(self.group,
                                                                self,
                                                                np.matmul(x.rep, self.rep),
                                                                self.current_chart)


class RepresentationLieGroupTangentVector(lgp.LieGroupTangentVector):

    def __init__(self,
                 group: RepresentationLieGroup,
                 configuration,
                 representation,
                 initial_chart=None,
                 initial_basis=0):
        """Tangent vector with extra group properties"""

        lgp.LieGroupTangentVector.__init__(self,
                                           group,
                                           configuration,
                                           group.identity_derep,  # This is to get something of the right shape, rep overrides whatever we have here
                                           initial_chart,
                                           initial_basis)

        self.rep = representation

        # Information about how to build a set of these objects
        self.plural = RepresentationLieGroupTangentVectorSet

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
            # If the representation is a matrix, assume that it is a proper representation
            pass
        elif representation.ndim == 1:
            # Multiply the matrices in the Jacobian of the representation function by the list of provided coefficients
            representation = np.matmul(self.group.representation_Jacobian_table[self.configuration.current_chart](self.configuration.value),
                                       representation)

        # Store the matrix representation
        self._representation = representation

    @property
    def value(self):

        val_raw = self.group.derepresentation_function_list[self.configuration.current_chart](self.rep)

        # Make sure that the value is a list or ndarray
        val = ut.ensure_ndarray(val_raw)

        return val

    @value.setter
    def value(self, val):

        # Pass the value input into the representation setter (which will force it to matrix form)
        self.rep = val


class RepresentationLieGroupElementSet(rgp.RepresentationGroupElementSet, lgp.LieGroupElementSet):
    """This is mostly a pass-through copy of representation group element set, but allows
    us to set the self.single attribute"""

    def __init__(self,
                 group,
                 representation=None,
                 initial_chart=0,
                 input_format=None):
        rgp.RepresentationGroupElementSet.__init__(self,
                                                   group,
                                                   representation,
                                                   initial_chart,
                                                   input_format)

        # Information about what this set should contain
        self.single = RepresentationLieGroupElement


class RepresentationLieGroupTangentVectorSet(lgp.LieGroupTangentVectorSet):
    pass
