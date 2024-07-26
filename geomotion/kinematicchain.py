from geomotion import utilityfunctions as ut, representationliegroup as rlgp, rigidbody as rb, plottingfunctions as gplt
import numpy as np

spot_color = gplt.crimson

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
                 proximal_position=G.identity_element(),
                 backdraft_position=G.identity_element()):
        ChainElement.__init__(self, transform, plot_info, proximal_position)

        # Save a default reference position
        self.reference_position = self.transform


class KinematicChain:

    def __init__(self,
                 links,
                 joints,
                 ground=None,
                 reparameterization_function=None):

        # Save the links, joints, ground, and reparameterization information
        self.links = links
        self.joints = joints
        self.ground = ground
        self.reparameterization_function = reparameterization_function

    def draw(self, ax, **kwargs):

        # Draw the links in the chain
        for l in self.links:
            if l.plot_info is not None:
                l.draw(ax, **kwargs)

        # Draw the joints in the chain
        for j in self.joints:
            if j.plot_info is not None:
                j.draw(ax, **kwargs)

        # Draw a ground point if provided
        if self.ground is not None:
            self.ground.draw(ax)

    @property
    def link_centers(self):

        return rlgp.RepresentationLieGroupElementSet([l.medial_position for l in self.links])

    def parse_frame(self, framespec):

        # If the frame is specified via an integer, take that as a link index
        if isinstance(framespec, int):
            frame = self.links[framespec].position

        # Possible string specifications for frame:
        elif isinstance(framespec, str):

            # Place the frame half-way down the chain
            if framespec == 'midpoint':
                if len(self.links) % 2 != 0:
                    frame = self.links[len(self.links) // 2].medial_position
                else:
                    frame = self.joints[len(self.joints) // 2].medial_position

            # Place the frame at the proximal end of the proximal link
            elif framespec == 'proximal':
                frame = self.links[0].proximal_position


            # Place the frame at the center of mass and mean orientation (taking all links as equal mass for now)
            elif framespec == 'com':

                # Get a set of the link positions and extract to grid form
                link_pos = []
                for l in self.links:
                    link_pos.append(l.medial_position)
                link_pos_grid = link_pos[1].plural(link_pos).grid

                com = []
                for c in link_pos_grid:
                    com.append(np.mean(np.ravel(c)))
                frame = G.element(com)

            else:
                raise Exception("Unknown string used to specify frame on mobile chain")
        else:
            raise Exception("Unsupported format used to specify frame on mobile chain")

        return frame


class KinematicChainSequential(KinematicChain):
    def set_configuration(self,
                          joint_angles):

        # Push the provided coordinates through the reparameterization function if it exists
        if self.reparameterization_function is not None:
            joint_angles = self.reparameterization_function(joint_angles)

        for j, alpha in enumerate(joint_angles):
            self.joints[j].angle = alpha

        # Hand-load position of first Joint and first Link relative to base frame
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

    def set_configuration(self,
                          joint_angles):

        # Push the provided coordinates through the reparameterization function if it exists
        if self.reparameterization_function is not None:
            joint_angles = self.reparameterization_function(joint_angles)

        # Set the joint angles
        for j, alpha in enumerate(joint_angles):
            self.joints[j].angle = alpha

        exp_prod = G.identity_element()
        for j, joint in enumerate(self.joints):
            exp_prod = exp_prod * (joint.angle * joint.spatial_axis_ref).exp_L
            self.links[j].distal_position = exp_prod * self.links[j].reference_position

        self.joints[0].proximal_position = G.identity_element()
        for j, joint in enumerate(self.joints[1:], start=1):
            joint.proximal_position = self.links[j - 1].distal_position


class KinematicChainMobile(KinematicChain):

    def __init__(self,
                 links,
                 joints,
                 base='proximal',
                 baseframe_line_length=1,
                 reparameterization_function=None):

        # Initialize a kinematic chain with no ground
        KinematicChain.__init__(self, links, joints, None, reparameterization_function)
        self.position = G.identity_element()

        # Store the provided base-frame designator
        self.base = base
        self.baseframe_line_length = baseframe_line_length

        self.baseframe_indicator = rb.RigidBody(baseframe_line(baseframe_line_length))

    def draw(self, ax, baseframe_visible=True, **kwargs):

        # Draw the baseframe indicator if it is wanted
        if baseframe_visible:
            self.baseframe_indicator.draw(ax, **kwargs)

        # Draw all the links
        KinematicChain.draw(self, ax, **kwargs)

    def move_into_baseframe(self,
                            new_base,
                            old_base):

        old_frame = self.parse_frame(old_base)
        new_frame = self.parse_frame(new_base)

        rebase_transform = old_frame * new_frame.inverse

        for l in self.links:
            l.position = rebase_transform * l.position
        for j in self.joints:
            j.position = rebase_transform * j.position

        self.baseframe_indicator.position = self.position

    def rebase(self,
               new_base):

        # Take the current link and joint positions, and find what they would be if the position portion
        # of the frame configuration described the position of the new base frame
        self.move_into_baseframe(new_base, self.base)

        # Record the new base frame into the chain
        self.base = new_base


class KinematicChainMobileSequential(KinematicChainMobile):
    def set_configuration(self,
                          position,
                          joint_angles):

        # Push the provided coordinates through the reparameterization function if it exists
        if self.reparameterization_function is not None:
            joint_angles = self.reparameterization_function(joint_angles)

        # Store the system position and joint angles
        for j, alpha in enumerate(joint_angles):
            self.joints[j].angle = alpha
        self.position = position

        # Hand-load position of first link as if the system were being constructed using the proximal end as
        # a base frame
        self.links[0].proximal_position = position

        # Find positions of the rest of the chain elements
        for j, alpha in enumerate(joint_angles):
            self.joints[j].proximal_position = self.links[j].distal_position
            self.links[j + 1].proximal_position = self.joints[j].distal_position

        # Move the links and joints from their proximally-described positions to the specified base frame
        self.move_into_baseframe(self.base, 'proximal')


def ground_point(configuration, r, **kwargs):
    def T(body):
        return G.element_set(ut.GridArray([[0, 0, 0],
                                           [-r * np.sin(np.pi / 6), -r * np.cos(np.pi / 6), 0],
                                           [r * np.sin(np.pi / 6), -r * np.cos(np.pi / 6), 0]], 1),
                             0, "element")

    bar_width = 3 * r
    bar_offset = -r * np.cos(np.pi / 6)

    def bar(body):
        return G.element_set(ut.GridArray([[-bar_width / 2, bar_offset, 0],
                                           [bar_width / 2, bar_offset, 0]], 1))

    hash_height = .5 * r
    hash_angle = np.pi / 4
    hash_fraction = 0.9
    hash_tops = np.linspace(-bar_width / 2 + hash_height * np.tan(hash_angle), bar_width * (hash_fraction - .5), 4)

    hashes = []
    for ht in hash_tops:

        def hash_points(body, h=ht):

            return G.element_set(ut.GridArray([[h, bar_offset, 0],
                                               [h - hash_height * np.tan(hash_angle),
                                                bar_offset - hash_height,
                                                0]],
                                              1))

        hashes.append(hash_points)

        # Unpack the hashes and combine them with the triangle and bar
    plot_locus = [T, bar, *hashes]

    # Set the plot style
    plot_style = [{"edgecolor": 'black', "facecolor": 'white'} | kwargs] + \
                 [{"color": 'black', "zorder": -3}] * (len(plot_locus) + 1)

    # Set the plot function
    plot_function = ['fill'] + ['plot'] * (len(plot_locus) + 1)

    plot_info = rb.RigidBodyPlotInfo(plot_locus=plot_locus, plot_style=plot_style, plot_function=plot_function)

    return rb.RigidBody(plot_info, configuration)


def simple_link(r, spot_color='black', **kwargs):
    def L(body):
        return rb.SE2.element_set(
            ut.GridArray([[0, 0.05 * r, 0], [r, 0.05 * r, 0], [r, -0.05 * r, 0], [0, -0.05 * r, 0]], 1),
            0, "element")

    plot_locus = [L]

    plot_style = [{"edgecolor": 'black', "facecolor": 'white'} | kwargs]

    plot_info = rb.RigidBodyPlotInfo(plot_locus=plot_locus, plot_style=plot_style)

    return plot_info


def rotational_joint(l, **kwargs):
    def L(body):
        return G.element_set(ut.GridArray([[0, 0, 0], [l, 0, 0]], 1), 0, "element")

    plot_locus = [L]

    plot_style = [{"linestyle": 'dashed', "color": 'black', "zorder": -3} | kwargs]

    plot_function = ['plot']

    plot_info = rb.RigidBodyPlotInfo(plot_locus=plot_locus, plot_style=plot_style, plot_function=plot_function)

    return plot_info


def piston_link(length, backdraft_percent, width_ratio, **kwargs):
    def piston(body):
        return rb.SE2.element_set(ut.GridArray(
            [[-(backdraft_percent * length), width_ratio * length, 0], [length, width_ratio * length, 0],
             [length, -width_ratio * length, 0], [-(backdraft_percent * length), -width_ratio * length, 0]], 1),
            0, "element")

    plot_locus = [piston]

    plot_function = ['fill']

    plot_style = [{"edgecolor": 'black', "facecolor": 'white'} | kwargs]

    plot_info = rb.RigidBodyPlotInfo(plot_locus=plot_locus, plot_style=plot_style, plot_function=plot_function)

    return plot_info


def prismatic_joint(width, marklength, **kwargs):
    def reference(body):
        return rb.SE2.element_set(ut.GridArray([[0, width, 0], [0, width + marklength, 0]], 1),
                                  0, "element")

    def fiducial(body):
        return rb.SE2.element_set(ut.GridArray([[body.angle, width, 0], [body.angle, width + marklength, 0]], 1),
                                  0, "element")

    plot_locus = [reference, fiducial]

    plot_style = [{"linestyle": 'dashed', "color": 'black', "zorder": -3} | kwargs,
                  {"color": 'black', "zorder": -3} | kwargs]

    plot_function = ['fill', 'plot', 'plot']

    plot_info = rb.RigidBodyPlotInfo(plot_locus=plot_locus, plot_style=plot_style, plot_function=plot_function)

    return plot_info


def arc_link(length, radius, backdraft_percent, width_ratio, **kwargs):

    # Generate a set of points along the backbone of the arc
    centerline_feeds = np.linspace(-backdraft_percent * length, length, 30)

    # map the points into the world
    centerline = rb.SE2.element_set(ut.GridArray([radius * np.sin(centerline_feeds / radius),
                                                  radius * (1 - np.cos(centerline_feeds / radius)),
                                                  centerline_feeds / radius], 1),
                                    0,
                                    'component')

    # Generate top and bottom lines by applying each transform in the center line to points offset
    # above and below the line
    topline = centerline * rb.SE2.element([0, width_ratio * length, 0])
    bottomline = centerline * rb.SE2.element([0, - width_ratio * length, 0])

    def arc_piston(body):
        return rlgp.RepresentationLieGroupElementSet(topline.value + bottomline.value[::-1])

    plot_locus = [arc_piston]

    plot_function = ['fill']

    plot_style = [{"edgecolor": 'black', "facecolor": 'white'} | kwargs]

    plot_geometry = [{"radius": radius}]

    plot_info = rb.RigidBodyPlotInfo(plot_locus=plot_locus,
                                     plot_style=plot_style,
                                     plot_function=plot_function,
                                     plot_geometry=plot_geometry)

    return plot_info


def arc_joint(radius, width, marklength, **kwargs):
    def reference(body):
        return rb.SE2.element_set(ut.GridArray([[0, width, 0], [0, width + marklength, 0]], 1),
                                  0, "element")

    def fiducial(body):

        proximal_transform = rb.SE2.Lie_alg_vector([body.angle, 0, body.angle / radius]).exp_R

        return ( proximal_transform * reference(body))

    plot_locus = [reference, fiducial]

    plot_style = [{"linestyle": 'dashed', "color": 'black', "zorder": -3} | kwargs,
                  {"color": 'black', "zorder": -3} | kwargs]

    plot_function = ['fill', 'plot', 'plot']

    plot_info = rb.RigidBodyPlotInfo(plot_locus=plot_locus, plot_style=plot_style, plot_function=plot_function)

    return plot_info


def baseframe_line(l, **kwargs):
    def L(body):
        return G.element_set(ut.GridArray([[-l / 2, 0, 0], [l / 2, 0, 0]], 1), 0, "element")

    def D(body):
        return G.element_set(ut.GridArray([[0, 0, 0]], 1), 0, "element")

    plot_locus = [L, D]

    plot_function = ['plot', 'scatter']

    plot_style = [{"linestyle": 'dashed', "color": 'grey', "zorder": 3} | kwargs, {"color": spot_color, "zorder": 3}]

    plot_info = rb.RigidBodyPlotInfo(plot_locus=plot_locus, plot_style=plot_style, plot_function=plot_function)

    return plot_info
