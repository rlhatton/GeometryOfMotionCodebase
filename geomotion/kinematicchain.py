from geomotion import utilityfunctions as ut, representationliegroup as rlgp, rigidbody as rb
import numpy as np

G = rb.SE2


class ChainElement(rb.RigidBody):

    def __init__(self,
                 defining_geometry,
                 plot_info,
                 initial_configuration=G.identity_element()):

        # Set up the rigid body properties
        rb.RigidBody.__init__(self,
                              plot_info,
                              initial_configuration)

        # Make sure the defining geometry is of an appropriate form
        if isinstance(defining_geometry, rlgp.RepresentationLieGroupElement):
            self.transform = defining_geometry
        elif isinstance(defining_geometry, rlgp.RepresentationLieGroupTangentVector):
            self.vector = defining_geometry
        else:
            raise Exception("Defining geometry should be either a RepresentationLieGroupElement or "
                            "RepresentationLieGroupTangentVector")

        # Record the group for easy access
        self.group = G

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self,
                  h: rlgp.RepresentationLieGroupElement):
        # Store the transformation and its logarithm
        self._transform = h
        self._vector = h.log

    @property
    def vector(self):
        return self._vector

    @vector.setter
    def vector(self,
               v: rlgp.RepresentationLieGroupTangentVector):
        self._transform = v.exp_R
        self._vector = v

    def fractional_transform(self, frac):
        return (frac * self.vector).exp_R

    def fractional_position(self, frac):
        return self.proximal_position * self.fractional_transform(frac)

    @property
    def distal_position(self):
        return self.proximal_position * self.transform

    @distal_position.setter
    def distal_position(self, g):
        self.proximal_position = g * self.transform.inverse

    @property
    def medial_position(self):
        return self.proximal_position * self.fractional_transform(0.5)

    # Defining 'position' to be used with the RigidBody class
    @property
    def position(self):
        return self.proximal_position

    @position.setter
    def position(self, g):
        self.proximal_position = g


class Joint(ChainElement):

    def __init__(self,
                 local_axis: rlgp.RepresentationLieGroupTangentVector,
                 plot_info=None,
                 proximal_position=G.identity_element(),
                 angle=0):

        # Local axis should be at the group identity
        if all(np.isclose(local_axis.configuration.value, G.identity_element().value)):
            self.local_axis = local_axis
        else:
            raise Exception("Local axis vector must be at the group identity.")

        # Instantiate a chain element
        ChainElement.__init__(self, local_axis, plot_info, proximal_position)

        # Set a default value for the reference position
        self.reference_position = G.identity_element()

        # Set the joint angle
        self.angle = angle

    @property
    def spatial_axis(self):
        return self.proximal_position.Ad(self.local_axis)

    @property
    def spatial_axis_ref(self):
        return self.reference_position.Ad(self.local_axis)

    @property
    def world_axis(self):
        return self.proximal_position * self.local_axis

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, alpha):
        self._angle = alpha
        self.vector = alpha * self.local_axis


class Link(ChainElement):

    def __init__(self,
                 transform: G.element,
                 plot_info=None,
                 proximal_position=G.identity_element()):
        # A link is currently a basic chain element
        ChainElement.__init__(self, transform, plot_info, proximal_position)

        # Save a default reference position
        self.reference_position = self.transform


class KinematicChain:

    def __init__(self,
                 links,
                 joints,
                 ground=None):

        # Save the links, joints, and ground information
        self.links = links
        self.joints = joints
        self.ground = ground

    def draw(self, ax):

        # Draw the links in the chain
        for l in self.links:
            l.draw(ax)

        # Draw the joints in the chain
        for j in self.joints:
            j.draw(ax)

        # Draw a ground point if provided
        if self.ground is not None:
            self.ground.draw(ax)


class KinematicChainSequential(KinematicChain):
    def set_angles(self,
                   joint_angles):

        for j, alpha in enumerate(joint_angles):
            self.joints[j].angle = alpha

        # Hand-load position of first Joint
        self.joints[0].proximal_position = G.identity_element()
        self.links[0].proximal_position = self.joints[0].transform

        # Find positions of the rest of the chain elements
        for j, alpha in enumerate(joint_angles[1:], start=1):
            self.joints[j].proximal_position = self.links[j - 1].distal_position
            self.links[j].proximal_position = self.joints[j].distal_position


class KinematicChainPoE(KinematicChain):

    def __init__(self, links, joints):
        # Set up as a kinematic chain
        KinematicChain.__init__(self, links, joints)

        ###
        # Calculate the position with joint angles equal to zero

        for j, junk in enumerate(joints):
            self.joints[j].angle = 0

        # Hand-load position of first Joint
        self.joints[0].proximal_position = G.identity_element()
        self.links[0].proximal_position = self.joints[0].transform

        # Find positions of the rest of the chain elements,
        for j, x in enumerate(joints[1:], start=1):
            self.joints[j].proximal_position = self.links[j - 1].distal_position
            self.links[j].proximal_position = self.joints[j].distal_position

        # Save the proximal position of each joint as the reference position for the joint
        for joint in self.joints:
            joint.reference_position = joint.proximal_position

        # Save the distal positions for the links as their reference positions
        for link in self.links:
            link.reference_position = link.distal_position

    def set_angles(self,
                   joint_angles):

        # Set the joint angles
        for j, alpha in enumerate(joint_angles):
            self.joints[j].angle = alpha

        exp_prod = G.identity_element()
        for j, joint in enumerate(self.joints):
            exp_prod = exp_prod * (joint.angle * joint.spatial_axis_ref).exp_L
            self.links[j].distal_position = exp_prod * self.links[j].reference_position

        self.joints[0].proximal_position = G.identity_element()
        for j, joint in enumerate(self.joints[1:], start=1):
            joint.proximal_position = self.links[j-1].distal_position


def ground_point(configuration, r, **kwargs):
    T = G.element_set(ut.GridArray([[0, 0, 0],
                                    [-r * np.sin(np.pi / 6), -r * np.cos(np.pi / 6), 0],
                                    [r * np.sin(np.pi / 6), -r * np.cos(np.pi / 6), 0]], 1),
                      0, "element")

    bar_width = 3 * r
    bar_offset = -r * np.cos(np.pi / 6)
    bar = G.element_set(ut.GridArray([[-bar_width / 2, bar_offset, 0],
                                      [bar_width / 2, bar_offset, 0]], 1))

    hash_height = .5 * r
    hash_angle = np.pi / 4
    hash_fraction = 0.9
    hash_tops = np.linspace(-bar_width / 2 + hash_height * np.tan(hash_angle), bar_width * (hash_fraction - .5), 4)

    hashes = []
    for h in hash_tops:
        hashes.append(G.element_set(ut.GridArray([[h, bar_offset, 0],
                                                  [h - hash_height * np.tan(hash_angle), bar_offset - hash_height,
                                                   0]],
                                                 1)))

    # Unpack the hashes and combine them with the triangle and bar
    plot_points = [T, bar, *hashes]

    # Set the plot style
    plot_style = [{"edgecolor": 'black', "facecolor": 'white'} | kwargs] * len(plot_points)

    plot_info = rb.RigidBodyPlotInfo(plot_points=plot_points, plot_style=plot_style)

    return rb.RigidBody(plot_info, configuration)


def simple_link(r, spot_color='black', **kwargs):
    L = G.element_set(ut.GridArray([[0, 0.05, 0], [1, 0.05, 0], [1, -0.05, 0], [0, -0.05, 0]], 1),
                      0, "element")

    plot_points = [L]

    plot_style = [{"edgecolor": 'black', "facecolor": 'white'} | kwargs]

    plot_info = rb.RigidBodyPlotInfo(plot_points=plot_points, plot_style=plot_style)

    return plot_info


def joint_reference_line(l, **kwargs):
    L = G.element_set(ut.GridArray([[0, 0, 0], [l, 0, 0]], 1), 0, "element")

    plot_points = [L]

    plot_style = [{"linestyle": 'dashed', "color": 'black'} | kwargs]

    plot_function = 'plot'

    plot_info = rb.RigidBodyPlotInfo(plot_points=plot_points, plot_style=plot_style, plot_function=plot_function)

    return plot_info