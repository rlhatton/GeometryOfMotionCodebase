#! /usr/bin/python3
import numpy as np
from geomotion import group as gp

""" Affine addition """

def affine_addition(g_value,
                    h_value):

    return g_value + h_value

def affine_additive_inverse(g_value):

    return -g_value

R2plus = gp.Group(affine_addition, [0, 0], affine_additive_inverse)

g1 = R2plus.element([0, 1])
g3 = R2plus.element([1, 2])

g2 = g1.inverse_element() * g3

print("Pre-subtraction of " +str(g1.value) + " from " +str(g3.value) + " is " + str(g2.value))

""" Scalar multiplication """
def scalar_multiplication(g_value,
                          h_value):

    return g_value * h_value

def scalar_multiplicative_inverse(g_value):

    return 1/g_value

R1times = gp.Group(scalar_multiplication, [1], scalar_multiplicative_inverse)

g1 = R1times.element(2)
g3 = R1times.element(6)

g2 = g1.inverse_element() * g3

print("Pre-division of " +str(g1.value) + " from " +str(g3.value) + " is " + str(g2.value))

""" Modular addition """

def modular_addition(g,
                    h):
    return np.mod(g+h, 1)

def modular_additive_inverse(g_value):

    return np.mod(-g_value, 1)

S1plus = gp.Group(modular_addition, [0], modular_additive_inverse)

g1 = S1plus.element(.25)
g3 = S1plus.element(.125)

g2 = g1.inverse * g3

print("Modular pre-subtraction of " +str(g1.value) + " from " +str(g3.value) + " is " + str(g2.value))