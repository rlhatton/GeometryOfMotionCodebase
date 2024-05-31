#! /usr/bin/python3
import copy

import numpy as np
import numdifftools as ndt
import warnings
import manifold as md


class TangentBasis:

    """Class that stores a basis in the tangent space as a set of TangentVector elements"""

    def __init__(self,
                 vector_list,
                 configuration=None,
                 initial_basis=0,
                 initial_chart=0,
                 manifold: md.Manifold = None):

        # Check type of elements in vector_list, and convert them to vectors at the specified configuration if they
        # are not already TangentVectors

        # If vector_list is already a list of TangentVectors, do nothing
        if all(isinstance(vector_list[i], md.TangentVector) for i in vector_list):
            pass
        # If vector_list is a matrix or a list of vectors, and a configuration is provided, convert it to a list of
        # TangentVectors
        elif configuration is not None:

            # Make sure that vector_list is a list of ndarrays that can be interpreted as vectors

            # If vector_list is provided as an ndarray, separate it into a list of its columns
            if isinstance(vector_list, np.ndarray):
                vector_list = (vector_list[:][i] for i in vector_list)
            elif all(isinstance(vector_list[i], np.ndarray) for i in vector_list):
                pass
            else:
                raise Exception("Input vector_list is not a list of TangentVectors, a matrix, or a list of ndarrays "
                                "that can be interpreted as vectors")

            # Convert vector list to a list of TangentVectors
            vector_list = [md.TangentVector(vector_list[i], configuration, initial_basis, initial_chart, manifold)
                           for i in vector_list]

        else:
            raise Exception("Input vector_list is not a list of TangentVectors, but no configuration was provided at "
                            "which to construct the basis")

        # Save the list of TangentVector elements as the value of the basis
        self.value = vector_list

    def flatten(self):

        """Produce a square matrix in which each column is the value of the corresponding TangentVector in the basis"""
        flattened_form = np.concatenate((self.value[j] for j in range(self.manifold.n_dim)))

        return flattened_form

    @property
    def configuration(self):

        return (self.value[1]).configuration

    @property
    def manifold(self):

        return self.configuration.manifold

    def __mul__(self, other):
        """Multiplying a TangentBasis by a column array should produce a TangentVector whose value is the
         weighted sum of the basis elements"""
