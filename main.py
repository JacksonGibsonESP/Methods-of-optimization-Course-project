from sympy import *
import numpy as np
from numpy.linalg import inv

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

e = 0.000001
n = 9
d = 15
l1 = 5
l2 = 7
l3 = 10
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

vars = [a1, a2, a3, b1, b2, b3, c1, c2, c3]

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
    F_lambda.append(lambdify(vars, F[l], 'numpy'))

# print(sqrt(F_lambda[0](x0[0], x0[1], x0[2], x0[3], x0[4], x0[5], x0[6], x0[7], x0[8])))
# print(sqrt(F_lambda[1](x0[0], x0[1], x0[2], x0[3], x0[4], x0[5], x0[6], x0[7], x0[8])))
# print(sqrt(F_lambda[2](x0[0], x0[1], x0[2], x0[3], x0[4], x0[5], x0[6], x0[7], x0[8])))
# print(sqrt(F_lambda[3](x0[0], x0[1], x0[2], x0[3], x0[4], x0[5], x0[6], x0[7], x0[8])))
# print(sqrt(F_lambda[4](x0[0], x0[1], x0[2], x0[3], x0[4], x0[5], x0[6], x0[7], x0[8])))
# print(sqrt(F_lambda[5](x0[0], x0[1], x0[2], x0[3], x0[4], x0[5], x0[6], x0[7], x0[8])))
# print(sqrt(F_lambda[6](x0[0], x0[1], x0[2], x0[3], x0[4], x0[5], x0[6], x0[7], x0[8])))
# print(sqrt(F_lambda[7](x0[0], x0[1], x0[2], x0[3], x0[4], x0[5], x0[6], x0[7], x0[8])))
# print(sqrt(F_lambda[8](x0[0], x0[1], x0[2], x0[3], x0[4], x0[5], x0[6], x0[7], x0[8])))

F1 = [[] for i in range(n)]
F1_lambda = [[] for k in range(n)]

for i in range(n):
    for l in range(n):
        F1[i].append(F[i].diff(vars[l]))
        F1_lambda[i].append(lambdify(vars, F1[i][l], 'numpy'))

F2 = [[[] for k in range(n)] for i in range(n)]
F2_lambda = [[[] for k in range(n)] for i in range(n)]

for k in range(n):
    for i in range(n):
        for l in range(n):
            F2[k][i].append(F1[i][l].diff(vars[k]))
            F2_lambda[k][i].append(lambdify(vars, F2[k][i][l], 'numpy'))

F2 = np.array(F2, 'float')

x0 = np.array([0., 0., 5., 10., 17.32050807568877, 5., 20., 0., 5.], 'float')
iteration = 0

# p = 2
while True:
    F1_inv = [[] for i in range(n)]
    for i in range(n):
        for l in range(n):
            F1_inv[i].append(F1_lambda[i][l](x0[0], x0[1], x0[2], x0[3], x0[4], x0[5], x0[6], x0[7], x0[8]))

    F1_inv = np.array(F1_inv, 'float')
    F1_inv = inv(F1_inv)

    for i in range(n):
        F[i] = F_lambda[i](x0[0], x0[1], x0[2], x0[3], x0[4], x0[5], x0[6], x0[7], x0[8])

    N1 = np.dot(F1_inv, np.array(F, 'float'))
    N1 = N1.reshape(1, n)

    N1_2 = np.matmul(N1.reshape(n, 1), N1)

    N2 = np.matmul(F1_inv, (np.tensordot(F2, N1_2) * 0.5).reshape(9, 1)).reshape(1, n) + N1

    xp = np.subtract(x0, N2)

    iteration += 1
    diff = np.linalg.norm(np.subtract(xp, x0))
    print('%(iteration)d %(diff)g' % {"iteration": iteration, "diff": diff})

    if diff <= e:
        break
    x0 = xp.copy()[0]

xp = xp[0]
print(xp)

x = [A1, B1, C1, xp[0], xp[3], xp[6]]
y = [A2, B2, C2, xp[1], xp[4], xp[7]]
z = [0, 0, 0, xp[2], xp[5], xp[8]]
mpl.rcParams['legend.fontsize'] = 10
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.plot(x, y, z, 'o', label='tripod', linestyle='none')
ax.legend()
plt.plot([A1, B1], [A2, B2], [0, 0], color='k', linestyle='-', linewidth=2)
plt.plot([B1, C1], [B2, C2], [0, 0], color='k', linestyle='-', linewidth=2)
plt.plot([A1, C1], [A2, C2], [0, 0], color='k', linestyle='-', linewidth=2)
plt.plot([xp[0], xp[3]], [xp[1], xp[4]], [xp[2], xp[5]], color='k', linestyle='-', linewidth=2)
plt.plot([xp[3], xp[6]], [xp[4], xp[7]], [xp[5], xp[8]], color='k', linestyle='-', linewidth=2)
plt.plot([xp[0], xp[6]], [xp[1], xp[7]], [xp[2], xp[8]], color='k', linestyle='-', linewidth=2)
plt.plot([A1, xp[0]], [A2, xp[1]], [0, xp[2]], color='k', linestyle='-', linewidth=2)
plt.plot([B1, xp[3]], [B2, xp[4]], [0, xp[5]], color='k', linestyle='-', linewidth=2)
plt.plot([C1, xp[6]], [C2, xp[7]], [0, xp[8]], color='k', linestyle='-', linewidth=2)
plt.show()

x0 = np.array([0., 0., 5., 10., 17.32050807568877, 5., 20., 0., 5.], 'float')
iteration = 0

# p = 1
while True:
    F1_inv = [[] for i in range(n)]
    for i in range(n):
        for l in range(n):
            F1_inv[i].append(F1_lambda[i][l](x0[0], x0[1], x0[2], x0[3], x0[4], x0[5], x0[6], x0[7], x0[8]))

    F1_inv = np.array(F1_inv, 'float')
    F1_inv = inv(F1_inv)

    for i in range(n):
        F[i] = F_lambda[i](x0[0], x0[1], x0[2], x0[3], x0[4], x0[5], x0[6], x0[7], x0[8])

    N1 = np.dot(F1_inv, np.array(F, 'float'))

    xp = np.subtract(x0, N1)

    iteration += 1
    diff = np.linalg.norm(np.subtract(xp, x0))
    print('%(iteration)d %(diff)g' % {"iteration": iteration, "diff": diff})

    if diff <= e:
        break
    x0 = xp.copy()

print(xp)

# print(np.linalg.norm(np.subtract([xp[0], xp[1], xp[2]], [xp[3], xp[4], xp[5]])))
# print(np.linalg.norm(np.subtract([xp[3], xp[4], xp[5]], [xp[6], xp[7], xp[8]])))
# print(np.linalg.norm(np.subtract([xp[0], xp[1], xp[2]], [xp[6], xp[7], xp[8]])))
#
#
# print(sqrt(F_lambda[0](xp[0], xp[1], xp[2], xp[3], xp[4], xp[5], xp[6], xp[7], xp[8])))
# print(sqrt(F_lambda[1](xp[0], xp[1], xp[2], xp[3], xp[4], xp[5], xp[6], xp[7], xp[8])))
# print(sqrt(F_lambda[2](xp[0], xp[1], xp[2], xp[3], xp[4], xp[5], xp[6], xp[7], xp[8])))
# print(sqrt(F_lambda[3](xp[0], xp[1], xp[2], xp[3], xp[4], xp[5], xp[6], xp[7], xp[8])))
# print(sqrt(F_lambda[4](xp[0], xp[1], xp[2], xp[3], xp[4], xp[5], xp[6], xp[7], xp[8])))
# print(sqrt(F_lambda[5](xp[0], xp[1], xp[2], xp[3], xp[4], xp[5], xp[6], xp[7], xp[8])))
# print(sqrt(F_lambda[6](xp[0], xp[1], xp[2], xp[3], xp[4], xp[5], xp[6], xp[7], xp[8])))
# print(sqrt(F_lambda[7](xp[0], xp[1], xp[2], xp[3], xp[4], xp[5], xp[6], xp[7], xp[8])))
# print(sqrt(F_lambda[8](xp[0], xp[1], xp[2], xp[3], xp[4], xp[5], xp[6], xp[7], xp[8])))
