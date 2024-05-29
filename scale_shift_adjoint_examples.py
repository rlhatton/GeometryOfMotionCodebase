#! /usr/bin/python3
import numpy as np
import group as gp

def scale_shift_action(g_value,
                    h_value):

    gh_value = [g_value[0]*h_value[0], (g_value[0]*h_value[1])+g_value[1]]

    return gh_value

def scale_shift_inverse(g_value):

    g_inv_value = [1/g_value[0], -g_value[1]/g_value[0]]

    return g_inv_value

RxRplus = gp.Group(scale_shift_action, [1, 0], scale_shift_inverse)

g1 = RxRplus.element([3, -1])
g2 = RxRplus.element([.5, 1.5])

g3 = g1.adjoint_action(2)
g4 = g1.

print("Left scale-shift action of " +str(g1.value) + " on " +str(g2.value) + " is " + str(g3.value))

g4 = g2 * g1

print("Right scale-shift action of " +str(g1.value) + " on " +str(g2.value) + " is " + str(g4.value))

g1_inv = g1.inverse_element()
print("Inverse of g1 is " + str(g1_inv.value))

g_delta_right = g1_inv * g3
print("Left inverse scale-shift action of " +str(g1.value) + " on " +str(g3.value) + " is " + str(g_delta_right.value))