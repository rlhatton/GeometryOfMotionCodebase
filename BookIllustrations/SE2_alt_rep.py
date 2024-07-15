from geomotion import representationliegroup as rlgp
import numpy as np
import numdifftools as ndt


def rotmatrix(g):
    return np.array([[g[2] - g[0], g[1] - g[3]], [g[3] - g[1], g[2] - g[0]]])


def SE2_alt_rep(g_value):
    x1 = g_value[0]
    y1 = g_value[1]
    x2 = g_value[2]
    y2 = g_value[3]

    g_rep = [[x2 - x1, y1 - y2, x1],
             [y2 - y1, x2 - x1, y1],
             [0, 0, 1]]

    return g_rep


def SE2_alt_derep(g_rep):
    x1 = g_rep[0][2]
    y1 = g_rep[1][2]
    x2 = g_rep[0][0] + x1
    y2 = g_rep[1][0] + y1

    g_value = [x1, y1, x2, y2]

    return g_value


SE2_alt = rlgp.RepresentationLieGroup(SE2_alt_rep, [0, 0, 1, 0], SE2_alt_derep)


def main():
    e = SE2_alt.identity_element()

    g = SE2_alt.element([1, 0, 2, 0])

    g_g_inv = g * g.inverse
    g_inv_g = g.inverse * g

    print("Testing that group action and inverse satisfy identity rules:")

    print("e * e = ", e * e, " which is equal to e=", e, " so identity element passes first requirement\n")
    print("e * g = ", e * g, " which is equal to g * e=", g * e, "\n which is equal to g=", g,
          " so identity element passes second requirement\n")

    print("Inverse of g=", g, " is g_inv=", g.inverse, " for which g * g_inv = ", g_g_inv, "\n and g_inv * g =",
          g_inv_g,
          " are both equal to the identity")

    g2 = SE2_alt.element([0, 0, 1, 1])
    print("Square of an element with sqrt(2) length and at a 45 degree angle has length 2 at 90 degrees: ", g2 * g2)


if __name__ == '__main__':
    main()
