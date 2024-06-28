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
        return str(self.value)


class GeomotionSet(UserList):
    """ Generic class for sets of elements"""

    @property
    def shape(self):
        return ut.shape(self.value)

    @property
    def value(self):
        return self.data
