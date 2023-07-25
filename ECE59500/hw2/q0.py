# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 19:15:33 2020

@author: tbureete
"""

import cvxpy as cp
import numpy as np

m = 30
n = 20

np.random.seed(1)
A = np.random.randn(m, n)
b = np.random.randn(m)

x = cp.Variable(n)
objective = cp.Minimize(cp.sum_squares(A*x-b))
constraints = [0 <= x, x <= 1]
prob = cp.Problem(objective, constraints)

result = prob.solve()

print(x.value)

print(constraints[0].dual_value)

