from geomotion import rigidbody as rb, kinematicchain as kc
from matplotlib import pyplot as plt

G = rb.SE2


# Create a point at the identity
e = G.identity_element()

# Place a chain grounding point at the identity
gp = kc.ground_point(e, .25)

# Create a list of three rotational joints and three straight links
joints = []
links = []
rot_axis = G.Lie_alg_vector([0, 0, 1])
link_transform = G.element([1, 0, 0])
for j in range(3):
    joints.append(kc.Joint(rot_axis))
    links.append(kc.Link(link_transform, kc.simple_link(1)))

chain = kc.KinematicChainPoE(links, joints)
chain.set_angles([1, -1, 1])




ax = plt.subplot(1, 1, 1)
ax.set_aspect('equal')
gp.draw(ax)

for l in links:
    l.draw(ax)

plt.show()