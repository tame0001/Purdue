# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 17:44:49 2020

@author: tbureete
"""

import numpy as np
import matplotlib.pyplot as plt

scale = 0.01
x = np.arange(-3, 3+scale, scale)

y = np.exp(-0.5*np.power(x, 2))/(np.sqrt(2*np.pi))

plt.plot(x, y)
plt.savefig("hw1_2_b.pdf")