import geomotion.manifold as md
import geomotion.utilityfunctions as ut


def R1transshift(x):
    return x + 1


def R1transshiftinv(y):
    return y + 1


R1 = md.Manifold([[None, R1transshift], [R1transshiftinv, None]], 1)

q1 = R1.element(1)
q2 = R1.element(2)

q_set = md.ManifoldElementSet([q1, q2])
q_set_part_t = q_set.transition(ut.GridArray([0, 1], 1))

print(q_set_part_t[1])
print(q_set_part_t[1].current_chart)
