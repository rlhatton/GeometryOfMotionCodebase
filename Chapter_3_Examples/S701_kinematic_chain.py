from S700_Construct_SE2 import SE2, RigidBody, RigidBodyPlotInfo
from geomotion import utilityfunctions as ut
import numpy as np


def ground_point(configuration, r, **kwargs):
    T = SE2.element_set(ut.GridArray([[0, 0, 0],
                                      [-r * np.sin(np.pi / 6), -r * np.cos(np.pi / 6), 0],
                                      [r * np.sin(np.pi / 6), -r * np.cos(np.pi / 6), 0]], 1),
                        0, "element")

    bar_width = 3*r
    bar_offset = -r * np.cos(np.pi / 6)
    bar = SE2.element_set(ut.GridArray([[-bar_width/2, bar_offset, 0],
                                        [bar_width/2, bar_offset, 0]], 1))

    hash_height = .5*r
    hash_angle = np.pi/4
    hash_fraction = 0.9
    hash_tops = np.linspace(-bar_width/2 + hash_height*np.tan(hash_angle), bar_width*(hash_fraction-.5), 4)

    hashes = []
    for h in hash_tops:
        hashes.append(SE2.element_set(ut.GridArray([[h, bar_offset, 0],
                                                    [h-hash_height*np.tan(hash_angle), bar_offset-hash_height, 0]],
                                                   1)))

    # Unpack the hashes and combine them with the triangle and bar
    plot_points = [T, bar, *hashes]

    # Set the plot style
    plot_style = [{"edgecolor": 'black', "facecolor": 'white'} | kwargs] * len(plot_points)

    plot_info = RigidBodyPlotInfo(plot_points=plot_points, plot_style=plot_style)

    return RigidBody(plot_info, configuration)


