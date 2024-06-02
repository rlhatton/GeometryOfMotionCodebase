import numpy as np


def vfun(p):
    print("input to vfun is " + str(p))
    return np.array([[p[0]], [p[1]]])


# points described as a list of arrays, each array containing one grid coordinate value at all points
outer_by_dimension_inner_by_gridpoint = [np.array([[-1, 1], [-1, 1]]), np.array([[1, 1], [-1, -1]])]
#print(outer_by_dimension_inner_by_gridpoint)

#outer_by_gridpoint_inner_by_dimension = np.transpose(outer_by_dimension_inner_by_gridpoint, [1, 2, 0])

outer_by_gridpoint_inner_by_dimension = np.array([outer_by_dimension_inner_by_gridpoint[0].flatten(), outer_by_dimension_inner_by_gridpoint[1].flatten()]).T

#print(outer_by_gridpoint_inner_by_dimension)

# Vectorize function
vfun_vec = np.vectorize(vfun)



target_structure = np.array([[[[-1, 1], [1,1] ], [[-1, -1], [1,-1]], ]])

print(target_structure)

print("Target structure shape is " + str(target_structure.shape))
print("pieces of target structure")
print(target_structure[0][0])

vectors_at_gridpoints = [[[vfun(target_structure[i][j][k]) for k in range(len(target_structure[i][j]))] for j in range(len(target_structure[i]))] for i in range(len(target_structure))]

#vectors_at_gridpoints = vfun_vec(target_structure)

#vectors_at_gridpoints = vfun_vec(outer_by_gridpoint_inner_by_dimension)
#
# print(vgrid)
#
#
# vgrid_vec = vfun_vec(pointgrid)
#
# print(vgrid_vec)
