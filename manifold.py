#! /usr/bin/python3
import copy

import numpy as np
import numdifftools as ndt
import warnings


class Manifold:
    """
    Class to hold manifold structure
    """

    def __init__(self,
                 transition_table,
                 n_dim):

        # Save the provided chart transition table as a class instance attribute
        self.transition_table = transition_table
        # Extract the number of charts implied by the transition table
        self.n_charts = len(transition_table)
        # Save the provided dimensionality as a class instance attribute
        self.n_dim = n_dim

        # Create a table of the Jacobians of the transition functions
        transition_Jacobian_table = [[[] for _ in range(self.n_charts)] for _ in range(self.n_charts)]
        for i in range(self.n_charts):
            for j in range(self.n_charts):
                if transition_table[i][j] is None:
                    transition_Jacobian_table[i][j] = None
                else:
                    transition_Jacobian_table[i][j] = ndt.Jacobian(transition_table[i][j])

        self.transition_Jacobian_table = transition_Jacobian_table

    def element(self,
                value,
                initial_chart=0):
        """Instantiate a manifold element with a specified value"""
        q = ManifoldElement(self,
                            value,
                            initial_chart)
        return q

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


class ManifoldElement:
    """
    Class for manifold elements
    """

    def __init__(self,
                 manifold,
                 value,
                 initial_chart=0):

        # Save the provided manifold, configuration, and initial chart as class instance attributes
        self.manifold = manifold
        # Make sure the value is a numpy float array
        self.value = np.array(value, dtype=float)
        self.current_chart = initial_chart

    def transition(self, new_chart):
        """
        Change the chart used to describe the manifold element, updating the numbers describing
        the configuration so that the actual point on the manifold stays the same
        """

        # Simple passthrough behavior if chart is not actually changing
        if new_chart == self.current_chart:

            new_value = self.value

        # Raise an exception if the transition from the current to new chart is not defined
        elif self.manifold.transition_table[self.current_chart][new_chart] is None:

            raise Exception(
                "The transition from " + str(self.current_chart) + " to " + str(new_chart) + " is undefined.")

        # If transition is non-trivial and defined, use the specified transition function
        else:

            new_value = self.manifold.transition_table[self.current_chart][new_chart](self.value)

        # Return a ManifoldElement with the new value and chart
        copied_element = copy.deepcopy(self)
        copied_element.value = new_value
        copied_element.current_chart = new_chart
        return copied_element


class TangentVector:

    def __init__(self,
                 value,
                 configuration,
                 initial_basis=0,
                 initial_chart=None,
                 manifold: Manifold = None):

        # If configuration is a manifold element, verify that no manifold was specified or that the configuration's
        # manifold matches the manifold specified for this vector
        if isinstance(configuration, ManifoldElement):
            if (manifold is None) or (configuration.manifold == manifold):
                pass
            else:
                raise Exception("Configuration specified for vector is not an element of the manifold to which the "
                                "vector is attached")
        # If configuration is not a manifold element, attempt to cast it to an np.array, check its size against the
        # manifold dimensionality, and if it is of the right size, use it to construct a configuration element of the
        # appropriate size in the appropriate chart
        elif manifold is not None:
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

        # Make sure that the value is a numpy array of floats, and that it is a column vector
        value = np.array(val, dtype=float)

        if value.shape[1] != 1:
            raise Exception(
                "Provided value is not a column vector. Make sure it is specified as a two-dimensional array with a "
                "single column")

        self._value = value


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
            # 'match' says to match the configuration to the new basis
            if configuration_transition == 'match':
                output_vector.configuration = matched_config
            # 'keep' says to leave the configuration as whatever it is currently
            elif configuration_transition == 'keep':
                pass
            else:
                raise Exception("Unknown option " + configuration_transition + "for transitioning the configuration "
                                                                               "while transitioning a TangentVector")
        else:
            # If a non-string was given, assume it identifies a specific chart to transition to
            output_vector.configuration = output_vector.configuration.transition(configuration_transition)

        return output_vector

    def vector_addition(self, other):

        # Verify that 'other' is a TangentVector
        if not isinstance(other, TangentVector):
            raise Exception("Cannot add TangentVector to an object of a different class")

        # Verify that 'other' is at the same configuration as self
        if self.configuration.manifold == other.configuration.manifold:
            if self.configuration.current_chart == other.configuration.current_chart:
                if np.isclose(self.configuration.value, other.configuration.value):
                    pass
                else:
                    raise Exception("Cannot add two TangentVectors at different configurations")
            else:
                # Convert other to the same chart as self and test for equality, but raise a warning
                if np.isclose(self.configuration.value,
                              (other.configuration.transition(self.configuration.current_chart)).value):
                    warnings.warn("TangentVectors have configurations described on  different charts, but appear to "
                                  "be at the same configuration")
                else:
                    raise Exception("Cannot add two TangentVectors at different configurations")
        else:
            raise Exception("Cannot add two TangentVectors attached to different manifolds")

        # Ensure that 'other' is expressed in the same basis as 'self'
        if self.current_basis == other.current_basis:
            pass
        else:
            warnings.warn("TangentVectors are expressed with respect to different bases, converting the second vector "
                          "into the basis of the first")
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
            output_vector = self.scalar_multiplication(1/other)
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
