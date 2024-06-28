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
               value,
               configuration,
               initial_basis=0,
               initial_chart=0):

        """Instantiate a tangent vector at a specified configuration on the manifold"""
        v = TangentVector(self,
                          value,
                          configuration,
                          initial_basis,
                          initial_chart)

        return v

    @property
    def vector_shape(self):

        """ Vectors should be 1-dimensional arrays with as many entries as there are dimensions
        This property facilitates checking this condition"""
        return (self.n_dim,)


class TangentVector(core.GeomotionElement):

    def __init__(self,
                 manifold: DiffManifold,
                 value,
                 configuration,
                 initial_basis=0,
                 initial_chart=None):

        # Make sure that the value is an ndarray
        value = ut.ensure_ndarray(value)

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
            configuration = ut.ensure_ndarray(configuration)
            if configuration.size == manifold.n_dim:
                # Use the initial chart, defaulting to the first chart
                if initial_chart is None:
                    initial_chart = 0
                # Generate the configuration as a manifold element at the specified configuration and chart
                configuration = manifold.element(configuration, initial_chart)
            else:
                raise Exception("Provided configuration coordinates do not match dimensionality of specified manifold")
        else:
            raise Exception("Manifold not specified and provided configuration does not have an associated manifold")

        # Check that the shape of the provided value matches the expected form
        if value.shape == self.manifold.element_shape:
            pass
        else:
            raise Exception("Value should be of shape ", self.manifold.vector_shape, " not ", value.shape)

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

        # Make sure that the value is an ndarray
        val = ut.ensure_ndarray(val)

        if len(val.shape) != 1:
            raise Exception(
                "Provided value is not a single-dimension array")

        self._value = val

    def __getitem__(self, item):
        return self.value[item]

    def __str__(self):
        return str(self.value)

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

        return self.__class__(self.manifold, new_value, output_configuration, new_basis, output_chart)

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
                              new_value,
                              self.configuration,
                              self.current_basis,
                              self.configuration.current_chart)

    def scalar_multiplication(self,
                              other):

        # Verify that 'other' is a scalar
        if not np.isscalar(other):
            raise Exception("Input for scalar multiplication is not a scalar")

        # Scale the value of the TangentVector value
        new_value = self.value * other

        return self.__class__(self.manifold,
                              new_value,
                              self.configuration,
                              self.current_basis,
                              self.configuration.current_chart)

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
                              new_value,
                              self.configuration,
                              self.current_basis,
                              self.configuration.current_chart)

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


class TangentVectorSet(md.GeomotionSet):

    def __init__(self, *args):

        n_args = len(args)

        # Check if the first argument is a list of TangentVectors, and if so, use it directly
        if isinstance(args[0], list):
            if ut.object_list_all_instance(TangentVector, args[0]):
                value = args[0]
                manifold = ut.object_list_extract_first_entry(args[0]).manifold
            else:
                raise Exception("List input to TangentVectorSet should contain TangentVector objects")

        # If the first argument is a manifold, process the inputs as if they were TangentVector inputs
        # provided in a GridArray
        elif isinstance(args[0], md.Manifold):
            manifold = args[0]
            element_shape = (manifold.n_dim,)
            if isinstance(args[1], ut.GridArray):

                # Extract the vector and configuration grid arrays from the argument list
                vector_grid = args[1]
                config_info = args[2]

                # Check if configuration is supplied as a single element for all vectors
                if isinstance(config_info, np.ndarray):
                    config_shape = config_info.shape
                    config_manifold_shape_match = (config_shape == (manifold.n_dim,))
                elif isinstance(config_info, list):
                    config_shape = ut.shape(config_info)
                    config_manifold_shape_match = (config_shape == [manifold.n_dim])
                elif isinstance(config_info, md.ManifoldElement):
                    config_manifold_shape_match = None
                else:
                    raise Exception("Supplied configuration information is not an ndarray, list, or ManifoldElement.")

                if isinstance(config_info, md.ManifoldElement) or config_manifold_shape_match:
                    single_configuration = True
                else:
                    single_configuration = False

                # Check if the format of the grids has been specified
                if n_args > 5:
                    input_format = args[5]
                else:
                    input_format = None

                # Make sure that the vector component grid is in element-outer format
                vector_grid = ut.format_grid(vector_grid, element_shape, 'element', input_format)

                if not single_configuration:

                    # Format the configuration grid
                    config_info = ut.format_grid(config_info,
                                                 (manifold.n_dim,),
                                                 'element',
                                                 input_format)

                    # Verify that the element-wise configuration grid is of matching dimension to the vector grid
                    if vector_grid.shape[:vector_grid.n_outer] == config_info.shape[:config_info.n_outer]:
                        pass
                    else:
                        raise Exception("Vector grid and configuration grid do not have matching element-wise "
                                        "structures")

                # Convert element-outer grids to a list of TangentVectors, including passing any initial chart and
                # basis to the manifold element function
                if n_args > 3:
                    initial_basis = args[3]
                else:
                    initial_basis = 0

                if n_args > 4:
                    initial_chart = args[4]
                else:
                    initial_chart = 0

                # Call an appropriate construction function depending on whether we're dealing
                # one configuration across all vectors, or have paired vector and configuration
                # grids
                if single_configuration:

                    def tangent_vector_construction_function(vector_value):
                        return manifold.vector(vector_value,
                                               config_info,
                                               initial_basis,
                                               initial_chart)

                    value = ut.object_list_eval(tangent_vector_construction_function,
                                                vector_grid,
                                                vector_grid.n_outer)

                else:
                    def tangent_vector_construction_function(vector_value, configuration_value):
                        tangent_vector = manifold.vector(vector_value,
                                                         configuration_value,
                                                         initial_basis,
                                                         initial_chart)
                        return tangent_vector

                    value = ut.object_list_eval_pairwise(tangent_vector_construction_function,
                                                         vector_grid,
                                                         config_info,
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

        return vector_component_outer_grid_array, config_component_outer_grid_array

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
            output_tangent_vector = self.manifold.TangentVector(output_value,
                                                                self.configuration,
                                                                self.underlying_basis,
                                                                self.current_chart)
            return output_tangent_vector
        else:
            raise Exception("Vector basis has " + str(self.n_vectors) +
                            " elements, but array of coefficients is " + str(other.shape))


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
                return ut.ensure_ndarray(defining_function(q, t))

        elif len(sig.parameters) == 1:
            # noinspection PyUnusedLocal
            def def_function(q, t):
                return ut.ensure_ndarray(defining_function(q))
        else:
            raise Exception("Defining function should take either two (configuration, time) or one (configuration) "
                            "input")

        self.defining_function = def_function
        self.manifold = manifold
        self.defining_basis = defining_basis
        self.current_basis = defining_basis
        self.defining_chart = defining_chart
        self.current_chart = defining_chart

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
            configuration_value = ut.ensure_ndarray(configuration)

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

    def grid(self,
             configuration_grid: ut.GridArray,
             time=0,
             basis=None,
             chart=None,
             output_format='grid'):

        # If basis and chart are not specified, use tangent vector field's defining basis and chart
        if basis is None:
            basis = self.defining_basis
        if chart is None:
            chart = self.defining_chart

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

        # Perform the evaluation
        vectors_at_points = configuration_at_points.grid_eval(v_function)

        vector_grid = vectors_at_points.everse

        # Output format
        if output_format == 'grid':
            return vector_grid
        elif output_format == 'set':
            return TangentVectorSet(self.manifold, vector_grid, configuration_grid, basis, chart, 'component')

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
            v = TangentVector(self.manifold, u, y, self.defining_basis, self.defining_chart)

            # Transition the vector into the new chart and basis
            v_new = v.transition(new_basis, configuration_transition)

            # extract the value from the transitioned vector
            u_new = v_new.value

            return u_new

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
