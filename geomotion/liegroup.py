#! /usr/bin/python3
import numpy as np
from . import utilityfunctions as ut
from . import manifold as md
from . import group as gp
from . import diffmanifold as tb
from operator import methodcaller


class LieGroup(gp.Group, tb.DiffManifold):

    def __init__(self,
                 operation_list,
                 identity_list,
                 inverse_function_list=None,
                 transition_table=((None,),)):

        gp.Group.__init__(self,
                          operation_list,
                          identity_list,
                          inverse_function_list,
                          transition_table)

        tb.DiffManifold.__init__(self,
                                 transition_table,
                                 self.n_dim)

    def vector(self,
               configuration,
               value,
               initial_chart=0,
               initial_basis=0):

        """Instantiate a Lie grouptangent vector at a specified configuration on the manifold"""
        v = LieGroupTangentVector(self,
                                  configuration,
                                  value,
                                  initial_chart,
                                  initial_basis)
        return v

    def vector_set(self,
                   configuration,
                   value=None,
                   initial_chart=0,
                   initial_basis=0,
                   input_grid_format=None):

        v = LieGroupTangentVectorSet(self,
                                     configuration,
                                     value,
                                     initial_chart,
                                     initial_basis,
                                     input_grid_format)
        return v


class LieGroupTangentVector(tb.TangentVector):

    def __init__(self,
                 manifold: LieGroup,
                 configuration,
                 value,
                 initial_chart=None,
                 initial_basis=0):

        super().__init__(manifold,
                         configuration,
                         value,
                         initial_chart,
                         initial_basis)

    @property
    def group(self):
        return self.manifold

    @group.setter
    def group(self, gp):
        self.manifold = gp

class LieGroupTangentVectorSet(tb.TangentVectorSet):

    def __init__(self,
                 group,  # Could also be a TangentVector, TangentVectorSet, or list of TangentVectors
                 configuration=None,
                 value=None,
                 initial_chart=0,
                 initial_basis=0,
                 input_grid_format=None):

        super().__init__(group,
                         configuration,
                         value,
                         initial_chart,
                         initial_basis,
                         input_grid_format)




