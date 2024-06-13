from geomotion import group as gp


def scale_shift_action(g_value,
                       h_value):
    gh_value = [g_value[0] * h_value[0], (g_value[0] * h_value[1]) + g_value[1]]

    return gh_value


def scale_shift_inverse(g_value):
    g_inv_value = [1 / g_value[0], -g_value[1] / g_value[0]]

    return g_inv_value


RxRplus = gp.Group(scale_shift_action, [1, 0], scale_shift_inverse)
