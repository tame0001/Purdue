# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 17:44:49 2020

@author: tbureete
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

x = np.random.normal(0, 1, 1000)

fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111)
ax1.hist(x, bins=4)
fig1.savefig("hw1_2_c_1.pdf")


fig2 =  plt.figure(2)
ax2 = fig2.add_subplot(111)
ax2.hist(x, bins=1000)
fig2.savefig("hw1_2_c_2.pdf")


mean, var = norm.fit(x)
print("mean = {:.5f} variance = {:.5f}".format(mean, var))

x_ref = np.linspace(-3, 3, num=1000)
ax3 = ax1.twinx()
ax3.plot(x_ref, norm.pdf(x_ref), 'r', label='fitted gaussian')
ax3.legend()
fig1.savefig("hw1_2_c_3.pdf")

ax4 = ax2.twinx()
ax4.plot(x_ref, norm.pdf(x_ref), 'r', label='fitted gaussian')
ax4.legend()
fig2.savefig("hw1_2_c_4.pdf")