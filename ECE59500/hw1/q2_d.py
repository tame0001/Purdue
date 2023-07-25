# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 19:52:52 2020

@author: tbureete
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

n = 1000
x = np.random.normal(0, 1, n)
m = np.linspace(1, 200, num=200)
amax = np.amax(x)
amin = np.amin(x)
h = (amax-amin)/m
j_hat = np.zeros(200)

for j in range(1, 201):
    p = np.zeros(j)
    for sample in x:
        try:
            p[(int((sample - amin)/h[j-1]))] += 1
        except IndexError: # Max value case
            p[j-1] += 1
            
    sum_p_square = np.sum(np.power(p/n, 2))
    j_hat[j-1] = (2 / (h[j-1] * (n-1)) - 
                  (((n+1) / (h[j-1]*(n-1))) * sum_p_square))

fig1 = plt.figure(1)
plt.plot(m, j_hat)
fig1.savefig("hw1_2_d_1.pdf")

m_star = np.argmin(j_hat) + 1
print(m_star)

fig2 =  plt.figure(2)
ax2 = fig2.add_subplot(111)
ax2.hist(x, bins=m_star)
x_ref = np.linspace(-3, 3, num=1000)
ax3 = ax2.twinx()
ax3.plot(x_ref, norm.pdf(x_ref), 'r', label='fitted gaussian')
ax3.legend()
fig2.savefig("hw1_2_d_2.pdf")