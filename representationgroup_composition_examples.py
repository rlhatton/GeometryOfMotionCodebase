#! /usr/bin/python3
import numpy as np
import representationgroup as rgp

""" Scalar addition """

def scalar_addition_rep(g_value):

    g_rep = [[1, g_value], [0, 1]]
    return g_rep


R1plus = rgp.RepresentationGroup(scalar_addition_rep, 0)

g1 = R1plus.element(2)
g2 = R1plus.element(3)

g3 = g1 * g2

print("Addition-group composition of " +str(g1.rep) + " and " +str(g2.rep) + " is " + str(g3.rep))

""" Scalar addition """

def scalar_multiplication_rep(g_value):

    g_rep = [[g_value]]
    return g_rep


R1times = rgp.RepresentationGroup(scalar_multiplication_rep, 1)

g1 = R1times.element(2.)
g2 = R1times.element(3.)

g3 = g1 * g2

print("Multiplication-group composition of " +str(g1.rep) + " and " +str(g2.rep) + " is " + str(g3.rep))

def modular_addition_rep(g_value):

    g_radians = 2*np.pi*g_value
    g_rep = [[np.cos(g_radians), -np.sin(g_radians)],
             [np.sin(g_radians),  np.cos(g_radians)]]
    return g_rep


S1plus = rgp.RepresentationGroup(modular_addition_rep, 0)

g1 = S1plus.element(.25)
g2 = S1plus.element(.875)

g3 = g1 * g2

print("Modular-addition-group composition of " +str(g1.rep) + " and " +str(g2.rep) + " is " + str(g3.rep))
