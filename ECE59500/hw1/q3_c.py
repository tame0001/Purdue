# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 03:22:24 2020

@author: tbureete
"""

import numpy as np
import matplotlib.pyplot as plt

n = 5000
mean = [0, 0]
cov = [[1, 0], [0, 1]]
x = np.random.multivariate_normal(mean, cov, n)

fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111)
plt.scatter(x[:,0], x[:,1])
ax1.set_xlabel('x1')
ax1.set_ylabel('x2').set_rotation(0)
fig1.savefig("hw1_3_c_1.pdf")  

a = np.array([[-1*np.sqrt(6)/2, -1*np.sqrt(2)/2],
              [-1*np.sqrt(6)/2, np.sqrt(2)/2]]) 
b = np.array([[2, 6]]) 
y = np.empty((0, 2))

for xi in x[:,]:
    y = np.append(y, (np.dot(a,xi))+b, axis=0)
    
fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111)
plt.scatter(y[:,0], y[:,1])
ax2.set_xlabel('y1')
ax2.set_ylabel('y2').set_rotation(0)
fig2.savefig("hw1_3_c_2.pdf")

cov_y = np.cov(y.T)
print("cov = ",cov_y)
mean = np.mean(y.T, axis=1)
print("mean = ",mean)
v, w = np.linalg.eig(cov_y)
print("v = ", v)
print("w = ", w)

