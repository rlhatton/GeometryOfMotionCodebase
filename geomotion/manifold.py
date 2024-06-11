#! /usr/bin/python3
import copy
import numpy as np
from operator import methodcaller
from collections import UserList

from . import utilityfunctions as ut


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

    def element(self,
                value,
                initial_chart=0):
        """Instantiate a manifold element with a specified value"""
        q = ManifoldElement(self,
                            value,
                            initial_chart)
        return q


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
        self.value = np.squeeze(np.array(value, dtype=float))
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


class ManifoldElementSet(UserList):

    def __init__(self, *args):

        n_args = len(args)

        # Check if the first argument is a list of ManifoldElements, and if so, use it directly
        if isinstance(args[0], list):
            if ut.object_list_all_instance(ManifoldElement, args[0]):
                value = args[0]
            else:
                raise Exception("List input to ManifoldElementSet should contain ManifoldElements")
        # If the first argument is a manifold, process the inputs as if they were ManifoldElement inputs
        # provided in a GridArray
        elif isinstance(args[0], Manifold):
            manifold = args[0]
            if isinstance(args[1], ut.GridArray):

                grid = args[1]

                # Test if GridArray format could be component-wise in the outer dimension and element-wise on the
                # inner dimensions
                c_outer_e_inner = (grid.n_outer == 1) and (grid.shape[0] == manifold.n_dim)

                # Test if GridArray format could be element-wise in the outer dimension and component-wise on the
                # inner dimensions
                e_outer_c_inner = (grid.n_inner == 1) and (grid.shape[-1] == manifold.n_dim)

                # Use results from these tests to determine grid format
                if c_outer_e_inner and e_outer_c_inner:

                    # Grid format is ambiguous. Check for fourth argument specifying which format is being used
                    if n_args > 3:
                        grid_format = args[3]
                        if grid_format == 'component':
                            grid = grid.everse
                        elif grid_format == 'element':
                            pass
                    else:
                        raise Exception("Grid format is ambiguous and a grid format was not provided")

                if c_outer_e_inner and (not e_outer_c_inner):
                    # Convert component-outer grid to element-outer grid
                    grid = grid.everse
                    #print("Detected component-outer grid and everted it")

                if (not c_outer_e_inner) and e_outer_c_inner:
                    # Keep element-outer grid
                    pass
                    #print("Detected element-outer grid and maintained it")

                if (not c_outer_e_inner) and (not e_outer_c_inner):
                    # Grid is not compatible with manifold structure
                    raise Exception("Grid does not appear to be a component-wise or element-wise grid compatible with "
                                    "the provided manifold")

                # Convert element-outer grid to a list of ManifoldElements
                def manifold_construction_function(x):
                    return manifold.element(x)
                value = ut.object_list_eval(manifold_construction_function, grid, grid.n_outer)
                print("object_list_eval returns: ", value)


            else:
                raise Exception(
                    "If first input to ManifoldElementSet is a Manifold, second input should be a GridArray")

        else:
            raise Exception("First argument to ManifoldElementSet should be either a list of "
                            "ManifoldElements or a Manifold")

        super().__init__(value)

        # # If the first argument is a list, test if it is a nested list of ManifoldElements
        # def manifold_element_test(x):
        #     assert isinstance(x, ManifoldElement), "List input to ManifoldElementSet should contain ManifoldElements"
        #
        # try:
        #     ut.object_list_eval(manifold_element_test, args[0], len(ut.shape(args[0])))
        # except:
        #     raise Exception()
        #
        # # If the value input is not an ndarray, assume it is a nested list of ManifoldElements
        # if not isinstance(value, np.ndarray):
        #self.value = args[0]

    @property
    def shape(self):
        return ut.shape(self.value)

    @property
    def value(self):
        return self.data

    def transition(self, new_chart):
        transition_method = methodcaller('transition', new_chart)

        new_set = ut.object_list_eval(transition_method,
                                      self.value)

        return self.__class__(new_set)
