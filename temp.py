# import numpy as np

# from scipy.special import comb
# from itertools import combinations

# class ReferenceVectors():
#     """Class object for reference vectors."""

#     def __init__(self, lattice_resolution, number_of_objectives):
#         """Create a simplex lattice."""
#         number_of_vectors = comb(
#             lattice_resolution + number_of_objectives - 1,
#             number_of_objectives - 1, exact=True)
#         temp1 = range(1, number_of_objectives + lattice_resolution)
#         temp1 = np.array(list(combinations(temp1, number_of_objectives-1)))
#         temp2 = np.array([range(number_of_objectives-1)]*number_of_vectors)
#         temp = temp1 - temp2 - 1
#         weight = np.zeros((number_of_vectors, number_of_objectives), dtype=int)
#         weight[:, 0] = temp[:, 0]
#         for i in range(1, number_of_objectives-1):
#             weight[:, i] = temp[:, i] - temp[:, i-1]
#         weight[:, -1] = lattice_resolution - temp[:, -1]
#         self.values = weight/lattice_resolution
#         self.number_of_objectives = number_of_objectives
#         self.lattice_resolution = lattice_resolution
#         self.number_of_vectors = number_of_vectors

#     def normalize(self):
#         norm = np.linalg.norm(self.values,axis=1)
#         norm=np.repeat(norm, self.number_of_objectives).reshape(
#             self.number_of_vectors,self.number_of_objectives)
#         self.values = np.divide(self.values,norm)

# u = ReferenceVectors(3,5)
# type(u.values)
# print(u.values)
# u.normalize()

# print(u.values)

import numpy as np

a = np.arange(1,20)
a = np.zeros((3,4), dtype=int)
print(a)
b = np.asarray([1,2,3,4])
print(b)
print(a+b)
a = a+b
a = a[0:2,0:-1]
print(a)