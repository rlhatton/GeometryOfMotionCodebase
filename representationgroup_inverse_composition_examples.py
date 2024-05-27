#! /usr/bin/python3
import numpy as np
import representationgroup as rgp

""" Scalar addition """

def scalar_addition_rep(g_value):

    g_rep = [[1, g_value], [0, 1]]
    return g_rep


R1plus = rgp.RepresentationGroup(scalar_addition_rep, 0)

g1 = R1plus.element(2)
g3 = R1plus.element(5)

g2 = g1.inverse_element() * g3

print("Addition-group inverse composition of " +str(g1.rep) + " with " +str(g3.rep) + " is " + str(g2.rep))

""" Scalar addition """

def scalar_multiplication_rep(g_value):

    g_rep = [[g_value]]
    return g_rep


R1times = rgp.RepresentationGroup(scalar_multiplication_rep, 1)

g1 = R1times.element(2.)
g3 = R1times.element(6.)

g2 = g1.inverse_element() * g3

print("Multiplication-group inverse composition of " +str(g1.rep) + " with " +str(g3.rep) + " is " + str(g2.rep))

def modular_addition_rep(g_value):

    g_radians = 2*np.pi*g_value
    g_rep = [[np.cos(g_radians), -np.sin(g_radians)],
             [np.sin(g_radians),  np.cos(g_radians)]]
    return g_rep


S1plus = rgp.RepresentationGroup(modular_addition_rep, 0)

g1 = S1plus.element(.25)
g3 = S1plus.element(.125)

g2 = g1.inverse_element() * g3

print("Modular-addition-group composition of " +str(g1.rep) + " and " +str(g2.rep) + " is " + str(g3.rep))