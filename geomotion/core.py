#! /usr/bin/python3
from collections import UserList
from . import utilityfunctions as ut


class GeomotionSet(UserList):
    """ Generic class for sets of elements"""

    @property
    def shape(self):
        return ut.shape(self.value)

    @property
    def value(self):
        return self.data
