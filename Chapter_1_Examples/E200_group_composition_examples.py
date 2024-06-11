#! /usr/bin/python3
import numpy as np
from geomotion import group as gp

""" Affine addition """


def affine_addition(g_value,
                    h_value):
    return g_value + h_value


R2plus = gp.Group(affine_addition, [0, 0])

g1 = R2plus.element([0, 1])
g2 = R2plus.element([1, 1])

g3 = g1 * g2

print("Sum of " + str(g1.value) + " and " + str(g2.value) + " is " + str(g3.value))

""" Scalar multiplication """


def scalar_multiplication(g_value,
                          h_value):
    return g_value * h_value


R1times = gp.Group(scalar_multiplication, [1])

g1 = R1times.element(2)
g2 = R1times.element(3)

g3 = g1 * g2

print("Product of " + str(g1.value) + " and " + str(g2.value) + " is " + str(g3.value))

""" Modular addition """


def modular_addition(g_value,
                     h_value):
    return np.mod(g_value + h_value, 1)


S1plus = gp.Group(modular_addition, [0])

g1 = S1plus.element(.25)
g2 = S1plus.element(.875)

g3 = g1 * g2

print("Modular sum of " + str(g1.value) + " and " + str(g2.value) + " is " + str(g3.value))
