#! /usr/bin/python3
import copy
import numpy as np
import numdifftools as ndt
import warnings
from inspect import signature
from typing import Union
from scipy.integrate import solve_ivp

from . import manifold as md
from . import utilityfunctions as ut


class DiffManifold(md.Manifold):
    """Class that instantiates differentiable manifolds"""

    def __init__(self,
                 transition_table,
                 n_dim):

        super().__init__(transition_table,
                         n_dim)

        # Create a table of the Jacobians of the transition functions
        transition_Jacobian_table = [[[] for _ in range(self.n_charts)] for _ in range(self.n_charts)]
        for i in range(self.n_charts):
            for j in range(self.n_charts):
                if transition_table[i][j] is None:
                    transition_Jacobian_table[i][j] = None
                else:
                    transition_Jacobian_table[i][j] = ndt.Jacobian(transition_table[i][j])

        self.transition_Jacobian_table = transition_Jacobian_table

    def vector(self,
               value,
               configuration,
               initial_basis=0,
               initial_chart=0):

        """Instantiate a tangent vector at the specified configuration on the manifold"""
        v = TangentVector(value,
                          configuration,
                          initial_basis,
                          initial_chart,
                          self)

        return v


class TangentVector:

    def __init__(self,
                 value,
                 configuration,
                 initial_basis=0,
                 initial_chart=None,
                 manifold: DiffManifold = None):

        # Make sure that the value and configuration are each a list or ndarray
        if not (isinstance(value, list) or isinstance(value, np.ndarray)):
            value = [value]

        if not (isinstance(configuration, list) or isinstance(configuration, np.ndarray)
                or isinstance(configuration, md.ManifoldElement)):
            configuration = [configuration]

        # If configuration is a manifold element, verify that no manifold was specified or that the configuration's
        # manifold matches the manifold specified for this vector
        if isinstance(configuration, md.ManifoldElement):
            if (manifold is None) or (configuration.manifold == manifold):
                self.manifold = configuration.manifold
            else:
                raise Exception("Configuration specified for vector is not an element of the manifold to which the "
                                "vector is attached")
        # If configuration is not a manifold element, attempt to cast it to an np.array, check its size against the
        # manifold dimensionality, and if it is of the right size, use it to construct a configuration element of the
        # appropriate size in the appropriate chart
        elif manifold is not None:
            self.manifold = manifold
            if np.size(np.array(configuration)) == manifold.n_dim:
                # Now that we know we're not pulling the chart from a provided ManifoldElement, make the default chart 0
                if initial_chart is None:
                    initial_chart = 0
                configuration = manifold.element(configuration, initial_chart)
            else:
                raise Exception("Provided configuration coordinates do not match dimensionality of specified manifold")
        else:
            raise Exception("Manifold not specified and provided configuration does not have an associated manifold")

        # Set the value, initial basis, and configuration for the vector
        self.value = value
        self.current_basis = initial_basis
        self.configuration = configuration

    @property
    def value(self):

        val = self._value
        return val

    @value.setter
    def value(self, val):

        # Make sure that the value and configuration are each a list or ndarray
        if not (isinstance(val, list) or isinstance(val, np.ndarray)):
            val = [val]

        # Make sure that the value is a numpy array of floats, and that it is a column vector
        value = np.array(val, dtype=float)

        if value.shape[1] != 1:
            raise Exception(
                "Provided value is not a column vector. Make sure it is specified as a two-dimensional array with a "
                "single column")

        self._value = value

    def __getitem__(self, item):
        return self.value[item]

    def __str__(self):
        return str(self.value)

    def transition(self,
                   new_basis,
                   configuration_transition='match'):

        # Get the current configuration in the coordinates that match the current coordinate basis
        matched_config = self.configuration.transition(self.current_basis)

        # Unless the transition is the trivial transition, get the Jacobian of the corresponding transition map and
        # multiply it by the current value
        if new_basis == self.current_basis:

            new_value = self.value

        else:

            transition_jacobian = self.configuration.manifold.transition_Jacobian_table[self.current_basis][new_basis]
            new_value = np.matmul(transition_jacobian(matched_config.value), self.value)

        # Make a copy of 'self', then replace the value and current basis
        output_vector = copy.deepcopy(self)
        output_vector.value = new_value
        output_vector.current_basis = new_basis

        # Transition the vector's configuration if called for
        if isinstance(configuration_transition, str):
            # 'match' says to match the configuration chart to the new basis
            if configuration_transition == 'match':
                output_configuration = matched_config
                output_chart = new_basis
            # 'keep' says to leave the configuration chart as whatever it is currently
            elif configuration_transition == 'keep':
                output_configuration = self.configuration
                output_chart = self.configuration.current_chart
            else:
                raise Exception("Unknown option " + configuration_transition + "for transitioning the configuration "
                                                                               "while transitioning a TangentVector")
        else:
            # If a non-string was given, assume it identifies a specific chart to transition to
            output_configuration = self.configuration.transition(configuration_transition)

        return self.__class__(new_value, output_configuration, new_basis, output_chart, self.manifold)

    def vector_addition(self, other):

        # Verify that 'other' is a TangentVector
        if not isinstance(other, TangentVector):
            raise Exception("Cannot add TangentVector to an object of a different class")

        # Verify that 'other' is at the same configuration as self
        if self.configuration.manifold == other.configuration.manifold:
            if self.configuration.current_chart == other.configuration.current_chart:
                if all(np.isclose(self.configuration.value, other.configuration.value)):
                    pass
                else:
                    raise Exception("Cannot add two TangentVectors at different configurations")
            else:
                # Test if configurations are equal when expressed in same chart
                if np.isclose(self.configuration.value,
                              (other.configuration.transition(self.configuration.current_chart)).value):
                    pass
                else:
                    raise Exception("Cannot add two TangentVectors at different configurations")
        else:
            raise Exception("Cannot add two TangentVectors attached to different manifolds")

        # Ensure that 'other' is expressed in the same basis as 'self'
        if self.current_basis == other.current_basis:
            pass
        else:
            # warnings.warn("TangentVectors are expressed with respect to different bases, converting the second vector "
            #               "into the basis of the first")
            other = other.transition(self.current_basis)

        # Add the values of the two TangentVectors together, and then create a clone of 'self' with the value the sum
        # of the converted values of 'self' and 'other'
        output_vector = copy.deepcopy(self)
        output_vector.value = self.value + other.value

        return output_vector

    def scalar_multiplication(self,
                              other):

        # Verify that 'other' is a scalar
        if not np.isscalar(other):
            raise Exception("Input for scalar multiplication is not a scalar")

        # Copy 'self', and scale the value by the scalar input
        output_vector = copy.deepcopy(self)
        output_vector.value = other * output_vector.value

        return output_vector

    def matrix_multiplication(self,
                              other):

        # Verify that 'other' is a matrix of the appropriate size
        if isinstance(other, np.ndarray):
            if len(other.shape) == 2:
                if other.shape[0] == other.shape[1]:
                    if other.shape[1] == self.value.shape[0]:
                        pass
                    else:
                        raise Exception("Input for matrix multiplication is a square matrix of the wrong size.")
                else:
                    raise Exception("Input for matrix multiplication is not square")
            else:
                raise Exception("Input for matrix multiplication is not two-dimensional")
        else:
            raise Exception("Input for matrix multiplication is not an ndarray")

        # Copy 'self', and multiply the matrix input into the vector value
        output_vector = copy.deepcopy(self)
        output_vector.value = np.matmul(other, output_vector.value)

        return output_vector

    def __add__(self, other):

        return self.vector_addition(other)

    def __radd__(self, other):

        return self.vector_addition(other)

    def __mul__(self, other):

        # Scalar multiplication
        if np.isscalar(other):
            output_vector = self.scalar_multiplication(other)
        # Vector multiplication
        elif isinstance(other, np.ndarray):
            raise Exception("Undefined __mul__ behavior for TangentVector acting on matrices")
        # Undefined interaction
        else:
            raise Exception("Undefined __mul__ behavior for TangentVector acting on " + type(other))

        return output_vector

    def __rmul__(self, other):

        # Scalar multiplication
        if np.isscalar(other):
            output_vector = self.scalar_multiplication(other)
        # Vector multiplication
        elif isinstance(other, np.ndarray):
            output_vector = self.matrix_multiplication(other)
        # Undefined interaction
        else:
            raise Exception("Undefined __rmul__ behavior for TangentVector acting on " + type(other))

        return output_vector

    def __matmul__(self, other):

        # Vector multiplication
        if isinstance(other, np.ndarray):
            raise Exception("Undefined __matmul__ behavior for TangentVector acting on matrices")
        # Undefined interaction
        else:
            raise Exception("Undefined __mul__ behavior for TangentVector acting on " + type(other))

    def __rmatmul__(self, other):

        # Vector multiplication
        if isinstance(other, np.ndarray):
            output_vector = self.matrix_multiplication(other)
        # Undefined interaction
        else:
            raise Exception("Undefined __rmatmul__ behavior for TangentVector acting on " + type(other))

        return output_vector

    def __truediv__(self, other):
        # Scalar multiplication
        if np.isscalar(other):
            output_vector = self.scalar_multiplication(1 / other)
        # Vector multiplication
        elif isinstance(other, np.ndarray):
            raise Exception("Undefined __truediv__ behavior for TangentVector acting on matrices")
        # Undefined interaction
        else:
            raise Exception("Undefined __truediv__ behavior for TangentVector acting on " + type(other))

        return output_vector

    def __rtruediv__(self, other):

        # Scalar multiplication
        if np.isscalar(other):
            raise Exception("Undefined __rtruediv__ behavior for TangentVector acting on scalars")
        # Vector multiplication
        elif isinstance(other, np.ndarray):
            raise Exception("Undefined __rtruediv__ behavior for TangentVector acting on matrices")
        # Undefined interaction
        else:
            raise Exception("Undefined __rtruediv__ behavior for TangentVector acting on " + type(other))

class TangentVectorSet(ut.GeomotionSet)

class TangentBasis:
    """Class that stores a basis in a tangent space as a set of TangentVector elements"""

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

        # Verify that the configuration grid is a one-dimensional GridArray and that the dimensionality matches that
        # of the manifold
        if not isinstance(configuration_grid, ut.GridArray):
            raise Exception("Expected configuration_grid to be of type GridArray.")

        if configuration_grid.n_outer != 1:
            raise Exception("Expected n_outer to be 1 for the GridArray provided as configuration_grid.")

        if configuration_grid.shape[0] != self.manifold.n_dim:
            raise Exception("Expected the first axis of the GridArray provided as configuration_grid to match the "
                            "dimensionality of the manifold.")

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
        return self.integrate([t0, t0 + t_run], q0, 'final')
