#! /usr/bin/python3
import copy

import numpy as np
import numdifftools as ndt
import warnings
import manifold as md
from inspect import signature


class TangentBasis:
    """Class that stores a basis in the tangent space as a set of TangentVector elements"""

    def __init__(self,
                 vector_list,
                 configuration=None,
                 initial_underlying_basis=0,
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
            vector_list = [
                md.TangentVector(vector_list[i], configuration, initial_underlying_basis, initial_chart, manifold)
                for i in vector_list]

        else:
            raise Exception("Input vector_list is not a list of TangentVectors, but no configuration was provided at "
                            "which to construct the basis")

        # Save the list of TangentVector elements as the value of the basis
        self.vector_list = vector_list

    @property
    def matrix(self):

        """Produce a square matrix in which each column is the value of the corresponding TangentVector in the basis"""
        matrix_form = np.concatenate((self.vector_list[j].value for j in range(self.n_vectors)), 1)

        return matrix_form

    @property
    def configuration(self):
        return (self.vector_list[1]).configuration

    @property
    def manifold(self):
        return self.configuration.manifold

    @property
    def n_vectors(self):
        return len(self.vector_list)

    @property
    def underlying_basis(self):
        return (self.vector_list[1]).current_basis

    @property
    def current_chart(self):
        return self.configuration.current_chart

    def __mul__(self, other):
        """Multiplying a TangentBasis by a column array should produce a TangentVector whose value is the
         weighted sum of the basis elements"""

        # Attempt to cast other as an ndarray of floats
        other = np.array(other, dtype=float)

        if other.shape == (self.n_vectors, 1):
            output_value = np.matmul(self.matrix, other)
            output_tangent_vector = self.manifold.TangentVector(output_value,
                                                                self.configuration,
                                                                self.underlying_basis,
                                                                self.current_chart)
            return output_tangent_vector
        else:
            raise Exception("Vector basis has " + str(self.n_vectors) +
                            " elements, but array of coefficients is " + str(other.shape))

    def __add__(self, other):
        """Add two basis elements together if they are at the same location. Based on the rules encoded in the vector
        addition operation, the output will be in the chart and underlying basis of 'self'"""

        if self.n_vectors == other.n_vectors:

            # Add the corresponding vectors together
            output_vector_list = [self.vector_list[i] + other.vector_list[i] for i in range(self.n_vectors)]

            # Create a basis from the output vector list
            output_tangent_basis = TangentBasis(output_vector_list)

            return output_tangent_basis

        else:
            raise Exception("Cannot add vector bases with different numbers of elements")

    def transition(self, new_basis, configuration_transition='match'):
        """Apply transition to each vector"""

        output_vector_list = [self.vector_list[i].transition(new_basis, configuration_transition) for i in
                              range(self.n_vectors)]

        # Create a basis from the output vector list
        output_tangent_basis = TangentBasis(output_vector_list)

        return output_tangent_basis


class TangentVectorField:

    def __init__(self,
                 defining_function,
                 manifold,
                 defining_basis=0,
                 defining_chart=0):

        self.defining_function = defining_function
        self.manifold = manifold
        self.defining_basis = defining_basis
        self.defining_chart = defining_chart

    def evaluate_vector_field(self,
                              configuration,
                              time=0,
                              basis=None,
                              chart=None,
                              output_type='TangentVector'):

        # If basis and chart are not specified, use tangent vector field's defining basis and chart
        if basis is None:
            basis = self.defining_basis
        if chart is None:
            chart = self.defining_chart

        # If configuration is given as a full manifold element, extract its value
        if isinstance(configuration, md.ManifoldElement):
            configuration_value = configuration.value
        else:
            configuration_value = configuration

        # Evaluate the defining function
        defining_vector = self.defining_function(time, configuration_value)

        # Convert the coefficients returned by the defining function into a TangentVector
        defining_tangent_vector = self.manifold.TangentVector(defining_vector,
                                                              configuration,
                                                              self.defining_basis,
                                                              self.defining_chart,
                                                              self.manifold)

        # Transition the TangentVector into the specified chart and basis
        output_tangent_vector = defining_tangent_vector.transition(basis, chart)

        # Format output
        if output_type == 'TangentVector':
            pass
        elif output_type == 'array':
            output_tangent_vector = output_tangent_vector.value
        else:
            raise Exception("Unknown output type" + str(output_type))

        return output_tangent_vector

    def grid_evaluate_vector_field(self,
                                   configuration_grid,
                                   time=0,
                                   basis=None,
                                   chart=None):

        gridded_defining_function = np.vectorize(self.evaluate_vector_field, excluded=['time', 'basis', 'chart', 'output_type'])
        vectors_at_points = gridded_defining_function(configuration_grid, time, basis, chart, 'array')

        n_dim = self.manifold.n_dim
        new_ordering = np.concatenate((np.array([i for i in (range(n_dim, 2*n_dim))]), np.array([i for i in (range(0, n_dim))])))
        reordered_vector_grid = np.transpose(vectors_at_points, new_ordering)

        component_grids = [reordered_vector_grid[i][j] for i in reordered_vector_grid for j in reordered_vector_grid[i]]

        # # Extract the vector at the first point in the configuration grid
        # first_vector = tangent_vectors_at_points.flat(0)
        #
        # vector_grid = [np.empty(configuration_grid[0].shape, dtype=float) for i in range(len(first_vector.value))]
        # for

    def __call__(self,
                 configuration,
                 basis=None,
                 chart=None,
                 output_style=None):

        if isinstance(configuration, md.ManifoldElement):
            if output_style is None:
                output_style = 'rich'
        elif isinstance(configuration, np.ndarray):
            if output_style is None:
                output_style = 'matrix'


class TangentVectorFieldGrid:

    def __init__(self,
                 configuration_grid,
                 vector_grid):

        self.configuration_grid = configuration_grid,
        self.vector_grid = vector_grid,

