from SE2_alt import SE2_alt
from geomotion import liegroup as lgp

G = SE2_alt

# Create an identity group element and one shifted along the x axis by two units
e = G.identity_element()
g = G.element([2, 0, 3, 0])



# # Create a velocity vector in which the two points are both moving to the right at unit speed,
# # from which we can calculate the corresponding left and right groupwise velocities
# g_dot = G.vector(g, [1, 0, 1, 0])
# print("The left velocity corresponding to g_dot=", g_dot, " is g_circ_l=", g_dot.left)
# print("The right velocity corresponding to g_dot=", g_dot, " is g_circ_r=", g.inverse * g_dot)

# Create a configuration in which the link is rotated by 90 degrees, and both points
# are moving to the right at unit speed
g2 = G.element([0, 0, 0, 1])
g3 = G.element([0.1, 0, 0.1, 1])
g_dot = G.vector(g2, [1, 0, 1, 0])
print("The left velocity corresponding to g_dot=", g_dot, " is g_circ_l=", g_dot.left, " or ", g_dot.L_generator(g_dot.configuration))
print("The right velocity corresponding to g_dot=", g_dot, " is g_circ_r=", g.inverse * g_dot)

print(g2.inverse*g3)
print(G.R_generator([1, 0, 1, 0])(g2))
print(G.R_generator([1, 0, 1, 0])(lgp.LieGroupElementSet([g2]))[0])
