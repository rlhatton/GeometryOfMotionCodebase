#! /usr/bin/python3
import numpy as np
import representationgroup as rgp

""" Scalar addition """

def scalar_addition_rep(g_value):

    g_rep = [[1, g_value], [0, 1]]
    return g_rep

def scalar_addition_derep(g_rep):

    g_value = g_rep[0][1]
    return g_value


R1plus = rgp.RepresentationGroup(scalar_addition_rep, 0, scalar_addition_derep)

g1 = R1plus.element(2)
g2 = R1plus.element(3)

g3 = g1 * g2

print("Addition-group composition of " +str(g1.value) + " and " +str(g2.value) + " is " + str(g3.value))

""" Scalar addition """

def scalar_multiplication_rep(g_value):

    g_rep = [[g_value]]
    return g_rep

def scalar_multiplication_derep(g_rep):

    g_value = g_rep[0][0]
    return g_value

R1times = rgp.RepresentationGroup(scalar_multiplication_rep, 1, scalar_multiplication_derep)

g1 = R1times.element(2.)
g2 = R1times.element(3.)

g3 = g1 * g2

print("Multiplication-group composition of " +str(g1.value) + " and " +str(g2.value) + " is " + str(g3.value))

def modular_addition_rep_a(g_value):

    g_radians = 2*np.pi*g_value
    g_rep = [[np.cos(g_radians), -np.sin(g_radians)],
             [np.sin(g_radians),  np.cos(g_radians)]]
    return g_rep

def modular_addition_rep_b(g_value):

    g_radians = 2*np.pi*g_value
    g_rep = [[np.cos(g_radians), -np.sin(g_radians)],
             [np.sin(g_radians),  np.cos(g_radians)]]
    return g_rep

def modular_addition_derep_a(g_rep):

    g_value = np.arctan2(g_rep[1][0], g_rep[0][0]) / (2*np.pi)
    return g_value

def modular_addition_derep_b(g_rep):

    g_value = np.arctan2(g_rep[1][0], g_rep[0][0]) / (2*np.pi)
    if g_value < 0:
        g_value += 1

    return g_value

def tau_b_wrt_a(g_value):
    if g_value < 0:
        g_value += 1

def tau_a_wrt_b(g_value):
    if g_value > .5:
        g_value -= 1


transition_table = [[None, tau_b_wrt_a], [tau_a_wrt_b, None]]

rep_list = (modular_addition_rep_a,modular_addition_rep_b)
derep_list = (modular_addition_derep_a,modular_addition_derep_b)

S1plus = rgp.RepresentationGroup(rep_list, 0, derep_list, 0, transition_table)

g1 = S1plus.element(.25)
g2 = S1plus.element(.875)

g3 = g1 * g2

print("Modular-addition-group composition of " +str(g1.value) + " and " +str(g2.value) + " is " + str(g3.value))

g1a = g1.transition(1)
g2a = g2.transition(1)
g3a = g3.transition(1)

print("Modular-addition-group composition of " +str(g1a.value) + " and " +str(g2a.value) + " is " + str(g3a.value))