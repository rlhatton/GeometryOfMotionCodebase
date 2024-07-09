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
    """Class that instantiates differentiable manifolds. Changes from Manifold are:
    1. A transition Jacobian table is automatically generated from the transition table
    2. Vectors and vector sets can be spawned in the same manner as elements and element sets
    3. The shape of a vector element is generated from n_dim and saved for use by other functions checking size"""

    def __init__(self,
                 transition_table,
                 n_dim):

        # Initialize a manifold with the provided transition table and number of dimensions
        md.Manifold.__init__(self,
                             transition_table,
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

        v = TangentVector(self,
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

        """Instantiate a tangent vector set at a specified configuration on the manifold"""

        v = TangentVectorSet(self,
                             configuration,
                             value,
                             initial_chart,
                             initial_basis,
                             input_grid_format)

        return v

    @property
    def vector_shape(self):

        """ Vectors should be 1-dimensional arrays with as many entries as there are dimensions
        This property facilitates other functions checking this condition"""
        return (self.n_dim,)


class TangentVector(core.GeomotionElement):
    """Constructs a vector in the tangent space of a specified manifold at a specified point"""

    def __init__(self,
                 manifold: DiffManifold,
                 configuration,
                 value,
                 initial_chart=None,
                 initial_basis=0):

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

            # Test if configurations are equal when expressed in same chart
            if all(np.isclose(self.configuration.value,
                              (other.configuration.transition(self.configuration.current_chart)).value)):
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
                 manifold,  # Could also be a TangentVector, TangentVectorSet, or list of TangentVectors
                 configuration=None,
                 value=None,
                 initial_chart=0,
                 initial_basis=0,
                 input_grid_format=None):

        # Check if the first argument is a ManifoldElementSet already,
        # and if so, extract its value and manifold
        if isinstance(manifold, TangentVectorSet):
            tangent_vector_set_input = manifold
            value = tangent_vector_set_input.value
            manifold = tangent_vector_set_input.manifold

        # Check if the first argument is a bare TangentVector, and if so, wrap its value in a list
        elif isinstance(manifold, TangentVector):
            tangent_vector_input = manifold
            value = [tangent_vector_input]
            manifold = tangent_vector_input.manifold

        # Check if the first argument is a list of TangentVectors, and if so, use it directly
        elif isinstance(manifold, list):
            if ut.object_list_all_instance(TangentVector, manifold):
                tangent_vector_list_input = manifold
                value = tangent_vector_list_input
                manifold = ut.object_list_extract_first_entry(tangent_vector_list_input).manifold
            else:
                raise Exception("List input to TangentVectorSet should contain TangentVector objects")

        # If the first argument is a differentiable manifold, process the inputs as if they were
        # TangentVector inputs allowing for GridArray formatting
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
                        return manifold.vector(configuration,
                                               vector_value,
                                               initial_chart,
                                               initial_basis)

                    value = ut.object_list_eval(tangent_vector_construction_function,
                                                vector_grid,
                                                vector_grid.n_outer)

                else:
                    def tangent_vector_construction_function(configuration_value, vector_value):
                        tangent_vector = manifold.vector(configuration_value,
                                                         vector_value,
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
            raise Exception("First argument to TangentVectorSet should be a TangentVector, a list of "
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
                   configuration_transition='match'):

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
                 output_defining_basis=0,
                 output_chart=None,
                 output_basis=None):

        if not isinstance(defining_function, list):
            defining_function = [defining_function]

        if defining_chart is None:
            defining_chart = [0] * len(defining_function)
        elif not isinstance(defining_chart, list):
            defining_chart = [defining_chart]

        if output_defining_basis is None:
            output_defining_basis = [0] * len(defining_function)
        elif not isinstance(output_defining_basis, list):
            output_defining_basis = [output_defining_basis]

        # If a separate output chart and basis are not specified, match them to the defining chart and
        # output defining basis
        if output_chart is None:
            output_chart = defining_chart
        else:
            if not isinstance(output_chart, list):
                output_chart = [output_chart]

        if output_basis is None:
            output_basis = output_defining_basis
        else:
            if not isinstance(output_basis, list):
                output_basis = [output_basis]

        if not ut.shape(defining_function) == ut.shape(defining_chart):
            raise Exception("Defining function list and defining chart list do not have matching shapes")
        elif not ut.shape(defining_function) == ut.shape(output_defining_basis):
            raise Exception("Defining function list and output defining basis list do not have matching shapes")
        elif not ut.shape(defining_function) == ut.shape(output_chart):
            raise Exception("Defining function list and output chart list do not have matching shapes")
        elif not ut.shape(defining_function) == ut.shape(output_basis):
            raise Exception("Defining function list and output basis list do not have matching shapes")

        # Make dictionaries from chart designators to their positions in the lists of functions
        self.chart_dict = dict(zip(defining_chart, range(len(defining_chart))))
        self.output_defining_basis_dict = dict(zip(output_defining_basis, range(len(output_defining_basis))))
        self.output_chart_dict = dict(zip(output_chart, range(len(output_chart))))
        self.output_basis_dict = dict(zip(output_basis, range(len(output_basis))))

        # Make sure that the defining functions can take at least two inputs (configuration and time)
        def_function_list = []
        for i, f in enumerate(defining_function):
            sig = signature(f)
            if len(sig.parameters) == 1:
                # noinspection PyUnusedLocal
                def def_function(q, t, *args):
                    return ut.ensure_ndarray(f(q))
            else:
                def def_function(q, t, *args):
                    return ut.ensure_ndarray(f(q, t, *args))

            def_function_list.append(def_function)

        def postprocess_function_single(q, v, input_chart):
            output_vector = manifold.vector(q, v, defining_chart,
                                            self.output_defining_basis[self.output_defining_basis_dict[input_chart]]). \
                transition(self.output_basis[self.output_basis_dict[input_chart]],
                           self.output_chart[self.output_chart_dict[input_chart]])

            return output_vector

        def postprocess_function_multiple(q, v, input_chart):
            output_vector_set = manifold.vector_set(q, v, self.defining_chart, output_defining_basis).transition(
                output_basis,
                output_chart)
            return output_vector_set

        postprocess_function = [postprocess_function_single, postprocess_function_multiple]

        super().__init__(manifold,
                         def_function,
                         defining_chart,
                         postprocess_function)

        self.output_defining_basis = output_defining_basis
        self.output_chart = output_chart
        self.output_basis = output_basis

    def __call__(self,
                 config,
                 time=None,
                 *args,
                 **kwargs):

        if time is None:
            time = 0

        output = super().__call__(config, time, *args, **kwargs)
        return output

    def transition_output(self, new_output_basis, new_output_chart='match'):

        return self.__class__(self.manifold,
                              self.defining_function,
                              self.defining_chart,
                              self.output_defining_basis,
                              new_output_chart,
                              new_output_basis)

    def addition(self, other):

        # Verify that the other object is also a tangent vector field
        if isinstance(other, TangentVectorField):
            # Verify that the vector fields are associated with the same manifold
            if self.manifold == other.manifold:
                pass
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
                                       self.output_defining_basis)

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
                                                   self.output_defining_basis)

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

        # Verify that the initial configuration is a manifold element
        if not isinstance(initial_config, md.ManifoldElement):
            raise Exception("Initial configuration for vector flow should be a ManifoldElement")

        # Get the chart in which the initial configuration is specified
        initial_config_chart = initial_config.current_chart

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

            # Turn the current numerical value into a configuration element in the same chart
            # as the initial configuration
            x_config = self.manifold.element(x, initial_config_chart)

            # Evaluate the vector field at that location
            v = self.__call__(x_config, t)

            v_initial_chart = v.transition(initial_config_chart)

            # ravel required to match dimension of vector with dimension of state
            return np.ravel(v_initial_chart.value)

        sol = solve_ivp(flow_function,
                        timespan,
                        initial_config.value,
                        dense_output=True, **kwargs)

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


class DifferentialMap(md.ManifoldFunction):

    def __init__(self,
                 defining_map: md.ManifoldMap):

        self.manifold = defining_map.manifold
        self.defining_map = defining_map
        self.output_manifold = defining_map.output_manifold
        self.output_defining_chart = defining_map.output_defining_chart
        self.output_defining_basis = defining_map.output_defining_chart
        self.output_chart = defining_map.output_chart
        self.output_basis = defining_map.output_chart
        self.postprocess_function = [self.postprocess_function_single, self.postprocess_function_multiple]

        # Get the chart dictionaries from the defining map
        self.chart_dict = defining_map.chart_dict
        self.output_defining_chart_dict = defining_map.output_defining_chart_dict
        self.output_chart_dict = defining_map.output_chart_dict

    def postprocess_function_single(self, q, v, input_chart):
        v_defining_output_chart = self.output_manifold.vector(q,
                                                              v,
                                                              self.output_defining_chart[
                                                                  self.output_defining_chart_dict[input_chart]],
                                                              self.output_defining_basis[
                                                                  self.output_defining_chart_dict[input_chart]])

        v_output_chart = v_defining_output_chart.transition(self.output_chart[self.output_chart_dict[input_chart]],
                                                            self.output_basis[self.output_chart_dict[input_chart]])

        return v_output_chart

    def postprocess_function_multiple(self, q, v, input_chart):
        v_defining_output_chart = self.output_manifold.vector_set(q,
                                                                  v,
                                                                  self.output_defining_chart[
                                                                      self.output_defining_chart_dict[input_chart]],
                                                                  self.output_defining_basis[
                                                                      self.output_defining_chart_dict[input_chart]])

        v_output_chart = v_defining_output_chart.transition(self.output_chart[self.output_chart_dict[input_chart]],
                                                            self.output_basis[self.output_chart_dict[input_chart]])

        return v_output_chart

    def defining_map_numeric(self, q_numeric, chart, *args, **kwargs):
        q_manifold = self.manifold.element(q_numeric, chart)

        q_out_manifold = self.defining_map[self.chart_dict[chart]](q_manifold, *args, **kwargs)

        q_out_numeric = q_out_manifold.value

        return q_out_numeric

    def __call__(self, vector, *args, **kwargs):

        # This changes the name of the first argument in the call function, which is useful for
        # things like IDE hinting. ManifoldFunction has evolved from the first implementation to
        # the point where we could probably just copy the call function over and remove the inheritance
        return super().__call__(vector, *args, **kwargs)

    def preprocess(self, vector):
        """Input vector information is provided as one of:
            1. TangentVector
            3. TangentVectorSet

            and should be pushed to being a TangentVectorSet"""

        if isinstance(vector, TangentVector):
            value_type = 'single'
        elif isinstance(vector, TangentVectorSet):
            value_type = 'multiple'
        else:
            raise Exception("DifferentialMap must be called with a TangentVector or TangentVectorSet")

        # Push the vector input into TangentVectorSet form
        vector_set = TangentVectorSet(vector)

        # Get every point into a chart for which the map is defined
        def send_to_feasible_chart(v, chart_to_try=0):
            if v.configuration.current_chart in self.defining_map.defining_chart:
                return v.transition(v.current_chart)  # Matches the basis to the chart
            elif self.manifold.transition_table[v.configuration.current_chart][chart_to_try] is not None:
                return v.transition(chart_to_try)
            elif chart_to_try < len(self.manifold.transition_table):
                return send_to_feasible_chart(v, chart_to_try + 1)
            else:
                raise Exception("Point is not in a chart where the function is defined, "
                                "and does not have a transition to a chart in which the function is defined")
                # Two-step transitions are not checked yet; this is also where boundaries of charts could be checked

        vector_set = TangentVectorSet(ut.object_list_eval(send_to_feasible_chart, vector_set))

        # Get a list of the chart in which each vector is defined
        def extract_chart(v):
            return v.current_chart

        chart_grid_e = ut.GridArray(ut.object_list_eval(extract_chart, vector_set), n_inner=1)

        return vector_set, chart_grid_e, value_type

    def process(self, vector_set, chart_grid_e, *process_args, **kwargs):

        # Extract a numeric grid from the vector set
        config_grid_c, vector_grid_c = vector_set.grid

        config_grid_e = config_grid_c.everse
        vector_grid_e = vector_grid_c.everse

        def defining_map_with_inputs(q, chart):
            q_out = self.defining_map_numeric(q, chart, *process_args, **kwargs)

            return q_out

        # Evaluate the function over the configurations
        output_config_grid_e = ut.GridArray(ut.array_eval_pairwise(defining_map_with_inputs, config_grid_e,
                                                                   chart_grid_e, config_grid_e.n_outer),
                                            config_grid_e.n_outer)

        def diffdefining_function_with_inputs(q, v, chart):
            function_jacobian = ndt.Jacobian(lambda x: defining_map_with_inputs(x, chart))
            v_out = np.matmul(function_jacobian(q), v)

            return v_out

        # Evaluate the Jacobian at each point with the vector at that point
        output_vector_grid_e = ut.GridArray(ut.object_list_eval_threewise(diffdefining_function_with_inputs,
                                                                          config_grid_e,
                                                                          vector_grid_e,
                                                                          chart_grid_e,
                                                                          n_outer=config_grid_e.n_outer),
                                            n_outer=config_grid_e.n_outer)

        return output_config_grid_e, output_vector_grid_e

    def postprocess(self, configuration_grid, function_grid, value_type):

        vector_location_grid = function_grid[0]
        vector_value_grid = function_grid[1]

        if self.postprocess_function is not None:
            if value_type == 'single':
                vector_location = vector_location_grid[0]  # Extract the single output from the grid array
                vector_value = vector_value_grid[0]
                return self.postprocess_function[0](vector_location, vector_value)
            elif value_type == 'multiple':
                return self.postprocess_function[1](vector_location_grid, vector_value_grid)
            else:
                raise Exception("Value_type should be 'single' or 'multiple'.")
        else:
            return function_grid


class DirectionDerivative(TangentVectorField):

    def __init__(self,
                 defining_map: md.ManifoldMap):
        self.manifold = defining_map.manifold
        self.defining_map = defining_map
        self.defining_chart = defining_map.defining_chart
        self.output_manifold = defining_map.output_manifold
        self.output_defining_chart = defining_map.output_defining_chart
        self.output_defining_basis = defining_map.output_defining_chart
        self.output_chart = defining_map.output_chart
        self.output_basis = defining_map.output_chart
        self.postprocess_function = [self.postprocess_function_single, self.postprocess_function_multiple]

    def defining_map_numeric(self, q_numeric, delta, *args, **kwargs):
        q_manifold = self.manifold.element(q_numeric)

        q_out_manifold = self.defining_map(q_manifold, delta, *args, **kwargs)

        q_out_numeric = q_out_manifold.value

        return q_out_numeric

    def process(self, config_grid_e, *input_args, **kwargs):

        # This makes everything compatible with the vector field time input, and gives us a hook
        # in case time-varying directional derivative fields become important
        time = input_args[0]
        process_args = input_args[1:]

        def defining_map_with_inputs(config, delta):
            return self.defining_map_numeric(config, delta, *process_args, **kwargs)

        # Evaluate the function over the configurations
        def defining_map_with_inputs_zero(q):
            return defining_map_with_inputs(q, [0])

        # output_config_grid_e = config_grid_e.grid_eval(defining_map_with_inputs_zero)
        output_config_grid_e = config_grid_e

        def direction_deriv(config):
            def defining_map_at_config(d):
                return defining_map_with_inputs(config, d)

            return np.squeeze(ndt.Jacobian(defining_map_at_config)([0]))

        # Evaluate the defining function over the grid
        vector_grid_e = config_grid_e.grid_eval(direction_deriv)

        return output_config_grid_e, vector_grid_e

    def postprocess(self, configuration_grid, function_grid, value_type):

        vector_location_grid = function_grid[0]
        vector_value_grid = function_grid[1]

        if self.postprocess_function is not None:
            if value_type == 'single':
                vector_location = vector_location_grid[0]  # Extract the single output from the grid array
                vector_value = vector_value_grid[0]
                return self.postprocess_function[0](vector_location, vector_value)
            elif value_type == 'multiple':
                return self.postprocess_function[1](vector_location_grid, vector_value_grid)
            else:
                raise Exception("Value_type should be 'single' or 'multiple'.")
        else:
            return function_grid

    def postprocess_function_single(self, q, v):
        v_defining_output_chart = self.output_manifold.vector(q,
                                                              v,
                                                              self.output_defining_chart,
                                                              self.output_defining_basis)

        v_output_chart = v_defining_output_chart.transition(self.output_chart,
                                                            self.output_basis)

        return v_output_chart

    def postprocess_function_multiple(self, q, v):
        v_defining_output_chart = self.output_manifold.vector_set(q,
                                                                  v,
                                                                  self.output_defining_chart,
                                                                  self.output_defining_basis)

        v_output_chart = v_defining_output_chart.transition(self.output_chart,
                                                            self.output_basis)

        return v_output_chart
