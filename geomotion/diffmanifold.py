#! /usr/bin/python3
import numpy as np
import numdifftools as ndt
from operator import methodcaller
from inspect import signature
from typing import Union
from scipy.integrate import solve_ivp

from . import core
from . import manifold as md
from . import utilityfunctions as ut


class DiffManifold(md.Manifold):
    """Class that instantiates differentiable manifolds"""

    def __init__(self,
                 transition_table,
                 n_dim):

        # Initialize a manifold with the provided transition table and number of dimensions
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
               configuration,
               value,
               initial_chart=0,
               initial_basis=0):

        """Instantiate a tangent vector at a specified configuration on the manifold"""
        v = TangentVector(self, configuration, value, initial_chart, initial_basis)

        return v

    def vector_set(self,
                   configuration,
                   value=None,
                   initial_chart=0,
                   initial_basis=0,
                   input_grid_format=None):

        v = TangentVectorSet(self,
                             value,
                             configuration,
                             initial_chart,
                             initial_basis,
                             input_grid_format)

        return v

    @property
    def vector_shape(self):

        """ Vectors should be 1-dimensional arrays with as many entries as there are dimensions
        This property facilitates checking this condition"""
        return (self.n_dim,)


class TangentVector(core.GeomotionElement):

    def __init__(self,
                 manifold: DiffManifold,
                 configuration,
                 value,
                 initial_chart=None,
                 initial_basis=0):

        # # Make sure that the value is an ndarray
        # value = ut.ensure_ndarray(value)

        # If configuration is a manifold element, verify that no manifold was specified or that the configuration's
        # manifold matches the manifold specified for this vector
        if isinstance(configuration, md.ManifoldElement):

            if (manifold is None) or (configuration.manifold == manifold):
                self.manifold = configuration.manifold
            else:
                raise Exception("Configuration specified for vector is not an element of the manifold to which the "
                                "vector is attached")

        # If configuration is not a manifold element, set this TangentVector's manifold to the provided manifold,
        # then attempt to use the provided configuration to generate a manifold element
        elif manifold is not None:
            self.manifold = manifold
            configuration = manifold.element(configuration, initial_chart)
        else:
            raise Exception("Manifold not specified and provided configuration does not have an associated manifold")

        # Set the value, initial basis, and configuration for the vector
        self.current_basis = initial_basis
        self.configuration = configuration
        self.value = value  # Format is verified/ensured by the format_value method

    def format_value(self, val):

        # TangentVectors have the same 1-d data format as their associated ManifoldElements
        return self.configuration.format_value(val)

    @property
    def column(self):
        return ut.column(self.value)

    def transition(self,
                   new_basis,
                   configuration_transition='match'):

        # Unless the transition is the trivial transition, get the Jacobian of the corresponding transition map and
        # multiply it by the current value
        if new_basis == self.current_basis:

            new_value = self.value

        else:

            transition_jacobian = self.configuration.manifold.transition_Jacobian_table[self.current_basis][new_basis]
            new_value = np.matmul(transition_jacobian(self.configuration.value), self.value)

        # Transition the vector's configuration if called for
        if isinstance(configuration_transition, str):
            # 'match' says to match the configuration chart to the new basis
            if configuration_transition == 'match':
                output_configuration = self.configuration.transition(new_basis)
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
            output_chart = new_basis

        return self.__class__(self.manifold, output_configuration, new_value, output_chart, new_basis)

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
            # warnings.warn("TangentVectors are expressed with respect to different bases, converting the second
            # vector " "into the basis of the first")
            other = other.transition(self.current_basis)

        # Add the values of the two TangentVectors together
        new_value = self.value + other.value

        return self.__class__(self.manifold,
                              self.configuration,
                              new_value,
                              self.configuration.current_chart,
                              self.current_basis)

    def scalar_multiplication(self,
                              other):

        # Verify that 'other' is a scalar
        if not np.isscalar(other):
            raise Exception("Input for scalar multiplication is not a scalar")

        # Scale the value of the TangentVector value
        new_value = self.value * other

        return self.__class__(self.manifold,
                              self.configuration,
                              new_value,
                              self.configuration.current_chart,
                              self.current_basis)

    def matrix_multiplication(self,
                              other):

        # Verify that 'other' is a matrix of the appropriate size
        if isinstance(other, np.ndarray):  # Is a matrix
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

        # Multiply the matrix input into the column representation of the vector value, then ravel back to 1-d
        new_value = np.ravel(np.matmul(other, self.column))

        return self.__class__(self.manifold,
                              self.configuration,
                              new_value,
                              self.configuration.current_chart,
                              self.current_basis)

    def __add__(self, other):

        if isinstance(other, TangentVector) and (self.manifold == other.manifold):
            return self.vector_addition(other)
        else:
            return NotImplemented

    def __radd__(self, other):

        if isinstance(other, TangentVector) and (self.manifold == other.manifold):
            return self.vector_addition(other)
        else:
            return NotImplemented

    def __mul__(self, other):

        # Scalar multiplication
        if np.isscalar(other):
            return self.scalar_multiplication(other)
        # Undefined interaction
        else:
            return NotImplemented

    def __rmul__(self, other):

        # Scalar multiplication
        if np.isscalar(other):
            return self.scalar_multiplication(other)
        # Undefined interaction
        else:
            return NotImplemented

    def __matmul__(self, other):

        # Undefined interaction
        return NotImplemented

    def __rmatmul__(self, other):

        # Matrix-vector multiplication
        if isinstance(other, np.ndarray):
            return self.matrix_multiplication(other)
        # Undefined interaction
        else:
            return NotImplemented

    def __truediv__(self, other):
        # Scalar multiplication
        if np.isscalar(other):
            return self.scalar_multiplication(1 / other)
        # Undefined interaction
        else:
            return NotImplemented

    def __rtruediv__(self, other):

        return NotImplemented


class TangentVectorSet(core.GeomotionSet):

    def __init__(self,
                 manifold,  # Could also be a list of TangentVectors
                 configuration=None,
                 value=None,
                 initial_chart=0,
                 initial_basis=0,
                 input_grid_format=None):

        # n_args = len(args)

        # Check if the first argument is a list of TangentVectors, and if so, use it directly
        if isinstance(manifold, list):
            if ut.object_list_all_instance(TangentVector, manifold):
                value = manifold
                manifold = ut.object_list_extract_first_entry(value[0]).manifold
            else:
                raise Exception("List input to TangentVectorSet should contain TangentVector objects")

        # If the first argument is a differentiable manifold, process the inputs as if they were TangentVector inputs
        # allowing for GridArray formatting
        elif isinstance(manifold, DiffManifold):

            # Get the expected shape of the elements
            element_shape = manifold.vector_shape

            if isinstance(value, ut.GridArray):

                ######
                # Check if configuration is supplied as a single element for all vectors

                # Check if configuration information is an array of the same size as an element
                if isinstance(configuration, np.ndarray):
                    config_shape = configuration.shape
                    config_manifold_shape_match = (config_shape == manifold.element_shape)
                # Check if configuration is a list of the same size as an element
                elif isinstance(configuration, list):
                    config_shape = ut.shape(configuration)
                    config_manifold_shape_match = (config_shape == [manifold.n_dim])
                # Check if configuration is a ManifoldElement
                elif isinstance(configuration, md.ManifoldElement):
                    config_manifold_shape_match = None
                else:
                    raise Exception("Supplied configuration information is not an ndarray, list, or ManifoldElement.")

                # Use truth values from test to decide if there is a single configuration input
                if isinstance(configuration, md.ManifoldElement) or config_manifold_shape_match:
                    single_configuration = True
                else:
                    single_configuration = False

                # Make sure that the vector component grid is in element-outer format
                vector_grid = ut.format_grid(value, element_shape, 'element', input_grid_format)

                if not single_configuration:

                    # Format the configuration grid
                    config_grid = ut.format_grid(configuration,
                                                 manifold.element_shape,
                                                 'element',
                                                 input_grid_format)

                    # Verify that the element-wise configuration grid is of matching dimension to the vector grid
                    if vector_grid.shape[:vector_grid.n_outer] == config_grid.shape[:config_grid.n_outer]:
                        pass
                    else:
                        raise Exception("Vector grid and configuration grid do not have matching element-wise "
                                        "structures")
                else:
                    config_grid = None  # Avoids "config_grid not assigned warning" from separate if statements

                # Call an appropriate construction function depending on whether we're dealing
                # one configuration across all vectors, or have paired vector and configuration
                # grids
                if single_configuration:

                    def tangent_vector_construction_function(vector_value):
                        return manifold.vector(vector_value,
                                               configuration,
                                               initial_chart,
                                               initial_basis)

                    value = ut.object_list_eval(tangent_vector_construction_function,
                                                vector_grid,
                                                vector_grid.n_outer)

                else:
                    def tangent_vector_construction_function(configuration_value, vector_value):
                        tangent_vector = manifold.vector(vector_value,
                                                         configuration_value,
                                                         initial_chart,
                                                         initial_basis)
                        return tangent_vector

                    value = ut.object_list_eval_pairwise(tangent_vector_construction_function,
                                                         config_grid,
                                                         vector_grid,
                                                         vector_grid.n_outer)

            else:
                raise Exception(
                    "If first input to TangentVectorSet is a Manifold, second input should be a GridArray")

        else:
            raise Exception("First argument to ManifoldSet should be either a list of "
                            "TangentVectors or a Manifold")

        super().__init__(value)
        self.manifold = manifold

    @property
    def grid(self):
        def extract_value(x):
            return x.value

        # Get an array of the vector element values, and use nested_stack to make it an ndarray
        element_outer_grid = ut.nested_stack(ut.object_list_eval(extract_value,
                                                                 self.value))

        # Convert this array into a GridArray
        element_outer_grid_array = ut.GridArray(element_outer_grid, n_inner=1)

        vector_component_outer_grid_array = element_outer_grid_array.everse

        ################

        def extract_config(x):
            return x.configuration.value

        # Get an array of the manifold element values, and use nested_stack to make it an ndarray

        element_outer_grid = ut.nested_stack(ut.object_list_eval(extract_config,
                                                                 self.value))

        # Convert this array into a GridArray
        element_outer_grid_array = ut.GridArray(element_outer_grid, n_inner=1)

        config_component_outer_grid_array = element_outer_grid_array.everse

        return config_component_outer_grid_array, vector_component_outer_grid_array

    def vector_set_action(self, other, action_name):

        if hasattr(other, 'shape'):

            if self.shape == other.shape:

                new_set = ut.object_list_method_eval_pairwise(action_name, self.value, other.value)

            else:

                raise Exception("Cannot apply a set of TangentVectors to a set of a different size")

        else:

            action = methodcaller(action_name, other)
            new_set = ut.object_list_eval(action, self.value)

        return self.__class__(new_set)

    def transition(self,
                   new_basis,
                   configuration_transition):

        transition_method = methodcaller('transition', new_basis, configuration_transition)

        new_set = ut.object_list_eval(transition_method,
                                      self.value)

        return self.__class__(new_set)

    def vector_addition(self, other):
        return self.vector_set_action(other, 'vector_addition')

    def scalar_multiplication(self, other):
        return self.vector_set_action(other, 'scalar_multiplication')

    def matrix_multiplication(self, other):
        return self.vector_set_action(other, 'matrix_multiplication')

    def __add__(self, other):

        if ((isinstance(other, TangentVector) or isinstance(other, TangentVectorSet))
                and (self.manifold == other.manifold)):
            return self.vector_addition(other)
        else:
            return NotImplemented

    def __radd__(self, other):

        if ((isinstance(other, TangentVector) or isinstance(other, TangentVectorSet))
                and (self.manifold == other.manifold)):
            return self.vector_addition(other)
        else:
            return NotImplemented

    def __mul__(self, other):

        # Scalar multiplication
        if np.isscalar(other):
            return self.scalar_multiplication(other)
        # Undefined interaction
        else:
            return NotImplemented

    def __rmul__(self, other):

        # Scalar multiplication
        if np.isscalar(other):
            return self.scalar_multiplication(other)
        # Undefined interaction
        else:
            return NotImplemented

    def __matmul__(self, other):

        # Undefined interaction
        return NotImplemented

    def __rmatmul__(self, other):

        # Matrix-vector multiplication
        if isinstance(other, np.ndarray):
            return self.matrix_multiplication(other)
        # Undefined interaction
        else:
            return NotImplemented

    def __truediv__(self, other):
        # Scalar multiplication
        if np.isscalar(other):
            return self.scalar_multiplication(1 / other)
        # Undefined interaction
        else:
            return NotImplemented

    def __rtruediv__(self, other):

        return NotImplemented


class TangentBasis(TangentVectorSet):
    """Class that stores a basis in a tangent space as a TangentVectorSet whose vectors are all at the same point"""

    def __init__(self, *args):

        # Pass the inputs to the TangentVectorSet init function
        super().__init__(*args)

    @property
    def matrix(self):

        """Produce a matrix in which each column is the value of the corresponding TangentVector in the basis"""
        basis_matrix = np.concatenate([self.vector_list[j].column for j in range(self.n_vectors)], axis=1)

        return basis_matrix

    @property
    def configuration(self):
        return (self.vector_list[0]).configuration

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

        # Attempt to cast other as a 1-d ndarray of floats
        other = np.array(other, dtype=float)

        if other.shape == (self.n_vectors, 1):
            output_value = np.matmul(self.matrix, other)
            output_tangent_vector = self.manifold.vector(self.configuration,
                                                         self.value,
                                                         self.current_chart,
                                                         self.underlying_basis)
            return output_tangent_vector
        else:
            raise Exception("Vector basis has " + str(self.n_vectors) +
                            " elements, but array of coefficients is " + str(other.shape))


class TangentVectorField(md.ManifoldFunction):

    def __init__(self,
                 manifold: DiffManifold,
                 defining_function,
                 defining_chart=0,
                 defining_basis=0):

        # Make sure that the defining function can take at least two inputs (configuration and time)
        sig = signature(defining_function)
        if len(sig.parameters) == 1:
            # noinspection PyUnusedLocal
            def def_function(q, t, *args):
                return ut.ensure_ndarray(defining_function(q))
        else:
            def def_function(q, t, *args):
                return ut.ensure_ndarray(defining_function(q, t, *args))

        def postprocess_function_single(q, v):
            return manifold.vector(q, v, defining_chart, defining_basis)

        def postprocess_function_multiple(q, v):
            return manifold.vector_set(q, v, defining_chart, defining_basis)

        postprocess_function = [postprocess_function_single, postprocess_function_multiple]

        super().__init__(manifold,
                         def_function,
                         defining_chart,
                         postprocess_function)

        self.defining_basis = defining_basis

    def __call__(self,
                 config,
                 time=0,
                 *args,
                 **kwargs):

        return super().__call__(config, time, *args, **kwargs)

    def transition(self, new_basis, configuration_transition: Union[str, int] = 'match'):
        """Take a vector field defined in one basis and chart combination and convert it
         to a different basis and chart combination"""

        # Parse the configuration_transition option. It would be nice to just pass it into
        # the vector transition (pushforward), but we need to process it here for the pullback
        # of the vector field function itself
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
            # Pull back function through transition map

            # Convert configuration from new chart to current chart
            y = self.manifold.transition_table[new_chart][self.defining_chart](x)

            # Evaluate defining function in current chart
            u = self.defining_function(y, t)

            # Convert evaluated vector into a TangentVector
            v = TangentVector(self.manifold, y, u, self.defining_basis, self.defining_chart)

            # Transition the vector into the new chart and basis
            v_new = v.transition(new_basis, configuration_transition)

            # extract the value from the transitioned vector
            u_new = v_new.value

            return u_new

        # Create a TangentVectorField from the output_defining_function
        output_tangent_vector_field = self.__class__(self.manifold,
                                                     output_defining_function,
                                                     new_chart,
                                                     new_basis)

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
        sum_of_fields = self.__class__(self.manifold,
                                       sum_of_functions,
                                       self.defining_chart,
                                       self.defining_basis)

        return sum_of_fields

    def scalar_multiplication(self, other):
        # Verify that 'other' is a scalar
        if not np.isscalar(other):
            raise Exception("Input for scalar multiplication is not a scalar")

        # Define a function that has a scaled output from
        def scaled_defining_function(x, t):
            v = other * self.defining_function(x, t)

            return v

        scalar_product_with_field = self.__class__(self.manifold,
                                                   scaled_defining_function,
                                                   self.defining_chart,
                                                   self.defining_basis)

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
            v = self.__call__(x, t).value
            return np.ravel(v)  # required to match dimension of vector with dimension of state

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
