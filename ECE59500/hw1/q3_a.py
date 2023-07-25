# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 22:51:10 2020

@author: tbureete
"""

import numpy as np
import matplotlib.pyplot as plt

delta = 0.01
x1 = np.arange(-1, 5+delta, delta)
x2 = np.arange(0, 10+delta, delta)
X1, X2 = np.meshgrid(x1, x2)
exp = (2*np.power(X1, 2))+(2*np.power(X2, 2)-2*X1*X2+4*X1-20*X2+56)
Y = 1/(2*np.pi*np.sqrt(3))*np.exp(exp/-6)
fig = plt.figure()
ax1 = fig.add_subplot(111)
plt.contour(x1, x2, Y)
ax1.set_xlabel('x1')
ax1.set_ylabel('x2').set_rotation(0)
fig.savefig("hw1_3_a_1.pdf")