#! /usr/bin/python3
import copy

import numpy as np
import numdifftools as ndt
import warnings
import manifold as md
from inspect import signature
import utilityfunctions as ut
from typing import Union
from scipy.integrate import solve_ivp
from scipy.integrate import OdeSolution


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
        matrix_form = [self.vector_list[j].value for j in range(self.n_vectors)]

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

        # Make sure that the vector field can take both configuration and time as inputs
        sig = signature(defining_function)
        if len(sig.parameters) == 2:
            def def_function(q, t):
                return defining_function(q, t)

        elif len(sig.parameters) == 1:
            # noinspection PyUnusedLocal
            def def_function(q, t):
                return defining_function(q)
        else:
            raise Exception("Defining function should take either two (configuration, time) or one (configuration) "
                            "input")

        self.defining_function = def_function
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
            configuration_value = np.array(configuration, dtype=float)

        # Evaluate the defining function
        defining_vector = self.defining_function(configuration_value, time)

        # Convert the coefficients returned by the defining function into a TangentVector
        defining_tangent_vector = self.manifold.vector(defining_vector,
                                                       configuration,
                                                       self.defining_basis,
                                                       self.defining_chart)
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
                                   configuration_grid: ut.GridArray,
                                   time=0,
                                   basis=None,
                                   chart=None):

        # Take in data formatted with the outer grid indices corresponding to the dimensionality of the data and the
        # inner grid indices corresponding to the location of those data points

        # Verify that the configuration grid is one-dimensional
        if configuration_grid.n_outer != 1:
            raise Exception("Expected a GridArray with n_outer=1. Check the grid you are providing as input to this "
                            "function")

        # Convert the data grid so that the outer indices correspond the location of the data points and the inner
        # indices correspond to the dimensionality of the data
        configuration_at_points = configuration_grid.everse

        # Evaluate the defining function at each data location to get the vector at that location
        def v_function(x):
            v_at_x = self.evaluate_vector_field(x, time, basis, chart, 'array')
            return v_at_x

        vectors_at_points = configuration_at_points.grid_eval(v_function)

        # Convert the vector representation so that its outer indices correspond to the vector dimensions and the
        # inner indices the spatial location of the data
        vector_grid = vectors_at_points.everse

        return vector_grid

    def __call__(self,
                 configuration,
                 time=0,
                 basis=None,
                 chart=None,
                 output_style=None):

        if isinstance(configuration, md.ManifoldElement):
            if output_style is None:
                output_style = 'TangentVector'
        else:
            configuration = np.array(configuration)
            if output_style is None:
                output_style = 'array'

        vector_at_config = self.evaluate_vector_field(configuration, time, basis, chart)

        if output_style == 'TangentVector':
            return vector_at_config
        elif output_style == 'array':
            return vector_at_config.value
        else:
            raise Exception("Unknown output style " + output_style + "for vector field")

    def transition(self, new_basis, configuration_transition: Union[str, int] = 'match'):
        """Take a vector field defined in one basis and chart combination and convert it
         to a different basis and chart combination"""

        # Parse the configuration_transition option
        if isinstance(configuration_transition, str):
            # 'match' says to match the configuration to the new basis
            if configuration_transition == 'match':
                new_chart = new_basis
            elif configuration_transition == 'keep':
                new_chart = self.defining_chart
            else:
                raise Exception("Unknown option " + configuration_transition + "for transitioning the configuration "
                                                                               "while transitioning a "
                                                                               "TangentVectorField")
        else:
            new_chart = configuration_transition

        # Modify the defining function by pushing forward through the transition maps
        def output_defining_function(x, t):
            # pull back function through transition map
            y = self.manifold.transition_table[new_chart][self.defining_chart](x)
            u = self.defining_function(y, t)
            # push forward vector through transition map
            v = np.matmul(self.manifold.transition_Jacobian_table[self.defining_basis][new_basis](y), u)

            return v

        # Create a TangentVectorField from the output_defining_function
        output_tangent_vector_field = TangentVectorField(output_defining_function, self.manifold, new_basis, new_chart)

        return output_tangent_vector_field

    def addition(self, other):

        # Verify that the other object is also a tangent vector field
        if isinstance(other, TangentVectorField):
            # Verify that the vector fields are associated with the same manifold
            if self.manifold == other.manifold:
                # Bring other into the same basis nd chart if necessary
                if (self.defining_basis == other.defining_basis) and (self.defining_chart == other.defining_chart):
                    pass
                else:
                    other = other.transition(self.defining_basis, self.defining_chart)
            else:
                raise Exception("Cannot add vector fields associated with different manifolds")
        else:
            raise Exception("Cannot add a vector field to an object of another type")

        # Create a function that is the sum of the two vector field functions
        def sum_of_functions(x, t):
            sf = self.defining_function(x, t) + other.defining_function(x, t)
            return sf

        # Create a new TangentVectorField object
        sum_of_fields = TangentVectorField(sum_of_functions,
                                           self.manifold,
                                           self.defining_basis,
                                           self.defining_chart)

        return sum_of_fields

    def scalar_multiplication(self, other):
        # Verify that 'other' is a scalar
        if not np.isscalar(other):
            raise Exception("Input for scalar multiplication is not a scalar")

        # Define a function that has a scaled output from
        def scaled_defining_function(x, t):
            v = other * self.defining_function(x, t)

            return v

        scalar_product_with_field = TangentVectorField(scaled_defining_function,
                                                       self.manifold,
                                                       self.defining_basis,
                                                       self.defining_chart)

        return scalar_product_with_field

    def __add__(self, other):
        return self.addition(other)

    def __mul__(self, other):
        return self.scalar_multiplication(other)

    def __rmul__(self, other):
        return self.scalar_multiplication(other)

    def __sub__(self, other):
        return self + (-1 * other)

    def __rsub__(self, other):
        return (-1 * self) + other

    def __truediv__(self, other):
        return self * (1 / other)

    def integrate(self,
                  timespan,
                  initial_config,
                  output_content='sol',
                  output_format=None,
                  **kwargs):

        # Parse the output content and format specifications
        if output_content == 'sol':
            if output_format is None:
                output_format = 'array'
        elif output_content == 'final':
            if output_format is None:
                output_format = 'TangentVector'
        else:
            raise Exception("Unsupported output content: ", output_content)

        def flow_function(t, x):
            v = self.evaluate_vector_field(x, t, output_type='array')
            return np.squeeze(v)  # required to match dimension of vector with dimension of state

        sol = solve_ivp(flow_function, timespan, initial_config, dense_output=True, **kwargs)

        if output_content == 'sol':
            return sol
        else:
            q_history = ut.GridArray(sol.y, 1).everse
            q_final = q_history[-1]
            if output_format == 'TangentVector':
                q_final = self.manifold.element(q_final, self.defining_chart)

            return q_final

    def exp(self,
            q0,
            t0=0,
            t_run=1):

        """Shorthand for integration of flow"""
        return self.integrate([t0, t0+t_run], q0, 'final')

