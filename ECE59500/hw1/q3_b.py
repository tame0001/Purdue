# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 23:01:06 2020

@author: tbureete
"""

import numpy as np
import scipy

sigma = np.array([[2, 1],[1, 2]])
u, s, vh = np.linalg.svd(sigma, full_matrices=True)

sqrt_s = scipy.linalg.sqrtm(np.diag(s))
a = np.dot(u, sqrt_s)
print(a)