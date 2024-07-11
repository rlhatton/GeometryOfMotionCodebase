from geomotion import representationliegroup as rlgp


def scale_shift_rep(g_value):

    g_rep = [[g_value[0], g_value[1]], [0, 1]]

    return g_rep


def scale_shift_derep(g_rep):

    g_value = [g_rep[0][0], g_rep[0][1]]

    return g_value


RxRplus = rlgp.RepresentationLieGroup(scale_shift_rep, [1, 0], scale_shift_derep)
