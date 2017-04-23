from sympy import *

n = 9
d = 15
l1 = 5
l2 = 5
l3 = 5
A1 = -10
A2 = 0
B1 = 0
B2 = 10 * sqrt(3)
C1 = 10
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
f5 = (b1 - A1) ** 2 + (b2 - A2) ** 2 + b3 ** 2 - l2 ** 2
f6 = (c1 - A1) ** 2 + (c2 - A2) ** 2 + c3 ** 2 - l3 ** 2
f7 = (a1 - b1) ** 2 + (a2 - b2) ** 2 - (a1 - c1) ** 2 - (a2 - c2) ** 2
f8 = (b1 - a1) ** 2 + (b2 - a2) ** 2 - (b1 - c1) ** 2 - (b2 - c2) ** 2
f9 = (c1 - a1) ** 2 + (c2 - a2) ** 2 - (c1 - b1) ** 2 - (c2 - b2) ** 2

F = [f1, f2, f3, f4, f5, f6, f7, f8, f9]

F1 = []
F1_lambda = []

for i in range(9):
    for l in range(9):
        F1.append(F[i].diff(x[l]))
        print(F1[i * n + l])
        F1_lambda.append(lambdify(x, F1[i * n + l], 'numpy'))
    print()

# print(F1_lambda[0](1, 0, 0, 2, 0, 0, 0, 0, 0))
# print(F1[0].diff(x[0]))

F2 = []
F2_lambda = []

for k in range(9):
    for i in range(9):
        for l in range(9):
            F2.append(F1[i * n + l].diff(x[k]))
            print(F2[k * n ** 2 + i * n + l])
            F2_lambda.append(lambdify(x, F2[k * n ** 2 + i * n + l], 'numpy'))
        print()


