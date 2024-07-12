from geomotion import liegroup as lgp
import numpy as np

def rotmatrix(g):
    return np.array([[g[2]-g[0], g[1]-g[3]],[g[3]-g[1], g[2]-g[0]]]) #/L(g)

# def L(g):
#     return (np.sqrt(np.square(g[2]-g[0])+np.square(g[3]-g[1])))
def SE2_alt_action(g, h):

    x1y1 = np.array([[g[0]], [g[1]]])
    u1v1 = np.array([[h[0]], [h[1]]])
    u2v2 = np.array([[h[2]], [h[3]]])

    a1b1 = x1y1 + np.matmul(rotmatrix(g), u1v1)

    a2b2 = x1y1 + np.matmul(rotmatrix(g), u2v2)

    return np.concatenate(np.ravel(a1b1), np.ravel(a2b2), 0)

