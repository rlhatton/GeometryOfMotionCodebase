#! /usr/bin/python3
from collections import UserList
from . import utilityfunctions as ut


class GeomotionElement:
    """Generic class for manifold/group elements, (co)tangent vectors, etc"""

    # def __init__(self,
    #              value):
    #     self._value = None

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        self._value = self.format_value(val)

    def format_value(self, val):
        return ut.ensure_ndarray(val)

    def __getitem__(self, item):
        return self.value[item]

    def __str__(self):
        return str(self.value) # + 'G'


class GeomotionSet(UserList):
    """ Generic class for sets of elements"""

    @property
    def shape(self):
        return ut.shape(self.value)

    @property
    def value(self):
        return self.data


class PullbackFunction:
    """Base class for function pullbacks"""

    def __init__(self,
                 outer_function,
                 inner_function,
                 *outer_args,
                 **outer_kwargs):
        self.outer_function = outer_function
        self.inner_function = inner_function
        self.outer_args = outer_args
        self.outer_kwargs = outer_kwargs

    def __call__(self, *args, **kwargs):
        inner_eval = ut.ensure_tuple(self.inner_function(*args, **kwargs))

        outer_eval = self.outer_function(*inner_eval, *self.outer_args, *self.outer_kwargs)

        return outer_eval

    def transition(self, *args, **kwargs):

        if hasattr(self.inner_function, 'transition'):
            new_inner_function = self.inner_function.transition(*args, **kwargs)
            return self.__class__(self.outer_function,
                                  new_inner_function,
                                  *self.outer_args,
                                  **self.outer_kwargs)
        else:
            raise Exception("Inner function has no method 'transition'.")

    def transition_output(self, *args, **kwargs):

        if hasattr(self.outer_function, 'transition_output'):
            new_outer_function = self.outer_function.transition_output(*args, **kwargs)
            return self.__class__(new_outer_function,
                                  self.inner_function,
                                  *self.outer_args,
                                  **self.outer_kwargs)

    def pullback(self, other, *args, **kwargs):

        return self.__class__(self,
                              other,
                              *args,
                              **kwargs)
