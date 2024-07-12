from geomotion import liegroup as lgp
import numpy as np
import numdifftools as ndt


def rotmatrix(g):
    return np.array([[g[2] - g[0], g[1] - g[3]], [g[3] - g[1], g[2] - g[0]]])


def SE2_alt_action(g, h):
    x1y1 = np.array([[g[0]], [g[1]]])
    u1v1 = np.array([[h[0]], [h[1]]])
    u2v2 = np.array([[h[2]], [h[3]]])

    a1b1 = x1y1 + np.matmul(rotmatrix(g), u1v1)

    a2b2 = x1y1 + np.matmul(rotmatrix(g), u2v2)

    return np.concatenate([np.ravel(a1b1), np.ravel(a2b2)], 0)


def SE2_alt_inverse(g):
    x1y1 = np.array([[g[0]], [g[1]]])
    x2y2 = np.array([[g[2]], [g[3]]])

    R_inv = np.linalg.inv(rotmatrix(g))
    det_R_inv = np.linalg.det(R_inv)

    a1b1 = -np.matmul(R_inv, x1y1)

    a2b2 = (det_R_inv * np.array([g[2]-g[0], g[1]-g[3]])) - np.ravel(x1y1)

    return np.concatenate([np.ravel(a1b1), np.ravel(a2b2)], 0)

SE2_alt = lgp.LieGroup(SE2_alt_action, [0, 0, 1, 0], SE2_alt_inverse)

def main():


    e = SE2_alt.identity_element()

    g = SE2_alt.element([1, 0, 2, 0])

    g_g_inv = g * g.inverse
    g_inv_g = g.inverse * g

    print("Testing that group action and inverse satisfy identity rules:")

    print("e * e = ", e * e, " which is equal to e=", e, " so identity element passes first requirement\n")
    print("e * g = ", e * g, " which is equal to g * e=", g * e, "\n which is equal to g=", g,
          " so identity element passes second requirement\n")

    print("Inverse of g=", g, " is g_inv=", g.inverse, " for which g * g_inv = ", g_g_inv, "\n and g_inv * g =", g_inv_g,
          " are both equal to the identity")

    g2 = SE2_alt.element([0, 0, 1, 1])
    print("Square of an element with sqrt(2) length and at a 45 degree angle has length 2 at 90 degrees: ", g2*g2)

    print(ndt.Jacobian(lambda g: SE2_alt_action(g,[0, 0, 0, 1]))([0, 0, 1, 0]))


if __name__ == '__main__':
    main()