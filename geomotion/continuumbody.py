from geomotion import (utilityfunctions as ut,
                       diffmanifold as tb,
                       representationliegroup as rlgp,
                       rigidbody as rb,
                       plottingfunctions as gplt)
import numpy as np

spot_color = gplt.crimson

G = rb.SE2


class ContinuumBody(rb.RigidBody):

    def __init__(self,
                 shape_description_function,
                 s_span=(0, 1),
                 ground=None,
                 plot_info=None,
                 initial_position=G.identity_element()):
        # Set up the rigid body properties
        rb.RigidBody.__init__(self,
                              plot_info,
                              initial_position)

        # Record the group for easy access
        self.group = G

        self.ground = ground

        # Save the shape description function
        self.shape_description_function = shape_description_function
        self.s_span = s_span

        # Save an empty initial shape locus
        self.shape_locus = None

        # Save default width
        self.width = 0.03

    def set_configuration(self,
                          shape_parameters,
                          t=0):
        def backbone_flow_function(g_value, s):
            g = G.element(g_value)
            h_circ_f = G.Lie_alg_vector([1, 0, 0])
            h_circ_a = self.shape_description_function(shape_parameters, s, t)

            return (g * (h_circ_f + h_circ_a)).value

        backbone_flow_field = tb.TangentVectorField(rb.SE2, backbone_flow_function)

        self.shape_locus = backbone_flow_field.integrate(self.s_span, self.group.identity_element())

    def draw(self, ax, **kwargs):
        s_dense = np.linspace(self.s_span[0], self.s_span[1], 100)

        g_dense = G.element_set(ut.GridArray(self.shape_locus.sol(s_dense), 1), 0, 'component')

        s_coarse = np.linspace(self.s_span[0], self.s_span[1], 10)
        g_coarse = G.element_set(ut.GridArray(self.shape_locus.sol(s_coarse), 1), 0, 'component')

        topline = g_dense * rb.SE2.element([0, self.width / 2, 0])
        bottomline = g_dense * rb.SE2.element([0, - self.width / 2, 0])

        boundaryline = rlgp.RepresentationLieGroupElementSet(topline.value + bottomline.value[::-1])

        ax.fill(*boundaryline.grid[:2], facecolor='white', edgecolor='black')
        ax.scatter(*g_coarse.grid[:2], color=spot_color)

        # Draw a ground point if provided
        if self.ground is not None:
            self.ground.draw(ax)
