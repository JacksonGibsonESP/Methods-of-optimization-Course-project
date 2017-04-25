from sympy import *
import numpy as np
from numpy.linalg import inv

n = 9
d = 20
l1 = 5
l2 = 5
l3 = 5
A1 = 0
A2 = 0
B1 = 10
B2 = 17.32050807568877
C1 = 20
C2 = 0

a1 = Symbol('a1')
a2 = Symbol('a2')
a3 = Symbol('a3')
b1 = Symbol('b1')
b2 = Symbol('b2')
b3 = Symbol('b3')
c1 = Symbol('c1')
c2 = Symbol('c2')
c3 = Symbol('c3')

x = [a1, a2, a3, b1, b2, b3, c1, c2, c3]

f1 = (a1 - b1) ** 2 + (a2 - b2) ** 2 + (a3 - b3) ** 2 - d ** 2
f2 = (a1 - c1) ** 2 + (a2 - c2) ** 2 + (a3 - c3) ** 2 - d ** 2
f3 = (b1 - c1) ** 2 + (b2 - c2) ** 2 + (b3 - c3) ** 2 - d ** 2
f4 = (a1 - A1) ** 2 + (a2 - A2) ** 2 + a3 ** 2 - l1 ** 2
f5 = (b1 - B1) ** 2 + (b2 - B2) ** 2 + b3 ** 2 - l2 ** 2
f6 = (c1 - C1) ** 2 + (c2 - C2) ** 2 + c3 ** 2 - l3 ** 2
f7 = (a1 - B1) ** 2 + (a2 - B2) ** 2 - (a1 - C1) ** 2 - (a2 - C2) ** 2
f8 = (b1 - A1) ** 2 + (b2 - A2) ** 2 - (b1 - C1) ** 2 - (b2 - C2) ** 2
f9 = (c1 - A1) ** 2 + (c2 - A2) ** 2 - (c1 - B1) ** 2 - (c2 - B2) ** 2

F = [f1, f2, f3, f4, f5, f6, f7, f8, f9]
F_lambda = []

for l in range(n):
    F_lambda.append(lambdify(x, F[l], 'numpy'))

vec = [0, 0, 5, 10, 17.32050807568877, 5, 20, 0, 5]

print(sqrt(F_lambda[0](vec[0], vec[1], vec[2], vec[3], vec[4], vec[5], vec[6], vec[7], vec[8])))
print(sqrt(F_lambda[1](vec[0], vec[1], vec[2], vec[3], vec[4], vec[5], vec[6], vec[7], vec[8])))
print(sqrt(F_lambda[2](vec[0], vec[1], vec[2], vec[3], vec[4], vec[5], vec[6], vec[7], vec[8])))
print(sqrt(F_lambda[3](vec[0], vec[1], vec[2], vec[3], vec[4], vec[5], vec[6], vec[7], vec[8])))
print(sqrt(F_lambda[4](vec[0], vec[1], vec[2], vec[3], vec[4], vec[5], vec[6], vec[7], vec[8])))
print(sqrt(F_lambda[5](vec[0], vec[1], vec[2], vec[3], vec[4], vec[5], vec[6], vec[7], vec[8])))
print(sqrt(F_lambda[6](vec[0], vec[1], vec[2], vec[3], vec[4], vec[5], vec[6], vec[7], vec[8])))
print(sqrt(F_lambda[7](vec[0], vec[1], vec[2], vec[3], vec[4], vec[5], vec[6], vec[7], vec[8])))
print(sqrt(F_lambda[8](vec[0], vec[1], vec[2], vec[3], vec[4], vec[5], vec[6], vec[7], vec[8])))

F1 = [[] for i in range(n)]
F1_lambda = [[] for k in range(n)]

for i in range(n):
    for l in range(n):
        F1[i].append(F[i].diff(x[l]))
        F1_lambda[i].append(lambdify(x, F1[i][l], 'numpy'))

print(F1)

F1_inv = [[] for i in range(n)]
for i in range(n):
    for l in range(n):
        F1_inv[i].append(F1_lambda[i][l](vec[0], vec[1], vec[2], vec[3], vec[4], vec[5], vec[6], vec[7], vec[8]))

F1_inv = np.matrix(F1_inv, 'float')
print(F1_inv)
F1_inv = inv(F1_inv)
# print(F1_inv)

# F2 = []
# F2_lambda = []

# for k in range(n):
#     for i in range(n):
#         for l in range(n):
#             F2.append(F1[i * n + l].diff(x[k]))
#             print(F2[k * n ** 2 + i * n + l])
#             F2_lambda.append(lambdify(x, F2[k * n ** 2 + i * n + l], 'numpy'))
#         print()
