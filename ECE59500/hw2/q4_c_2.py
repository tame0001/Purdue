# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 00:41:54 2020

@author: tbureete
"""

import csv
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

male_train_data = []
with open("male_train_data.csv", "r") as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    for data in reader:
        male_train_data.append(data)
    male_train_data.pop(0)
csv_file.close()

female_train_data = []
with open("female_train_data.csv", "r") as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    for data in reader:
        female_train_data.append(data)
    female_train_data.pop(0)
csv_file.close()

male_train_data = np.asarray(male_train_data, dtype=np.float32)
male_train_data = np.delete(male_train_data, 0, 1)
one_arr = np.ones(male_train_data.shape[0])
one_arr = np.resize(one_arr, (male_train_data.shape[0], 1))
male_train_data = np.concatenate((male_train_data, one_arr), axis=1)

female_train_data = np.asarray(female_train_data, dtype=np.float32)
female_train_data = np.delete(female_train_data, 0, 1)
one_arr = np.ones(female_train_data.shape[0])
one_arr = np.resize(one_arr, (female_train_data.shape[0], 1))
female_train_data = np.concatenate((female_train_data, one_arr), axis=1)

A = np.concatenate((male_train_data, female_train_data), axis=0)
A[:, 1] = A[:, 1] / 100
b = np.concatenate((np.ones(male_train_data.shape[0]),
                    -1*np.ones(female_train_data.shape[0])), axis=0)
b = np.resize(b, (b.shape[0], 1))

lamb = 0.1
theta_lambda = np.dot(np.linalg.inv(np.dot(A.T, A) +
                                    lamb*np.eye(A.shape[1])),
                      np.dot(A.T, b))

epsilon_star = np.dot(A, theta_lambda) - b
epsilon_star = np.linalg.norm(epsilon_star)**2
epsilon_list = np.arange(0, 101, 1)
epsilon_list = epsilon_list*2 + epsilon_star

theta_list = []
norm_theta = []
norm_residual = []

for epsilon in epsilon_list:
    x = cp.Variable((A.shape[1],1))
    objective = cp.Minimize(cp.sum_squares(x))
    constraints = [cp.sum_squares(A*x-b) <= epsilon]
    prob = cp.Problem(objective, constraints)
    result = prob.solve(verbose=True)
    theta_list.append(x.value)
    norm_theta.append(np.linalg.norm(x.value)**2)
    residual = np.dot(A, x.value) - b
    norm_residual.append(np.linalg.norm(residual)**2)
   
print(np.argmin(norm_residual))
print(theta_list[np.argmin(norm_residual)])


fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(norm_theta, norm_residual)
ax1.set_xlabel(r'$||{\theta}_{\epsilon } ||_2^2$')
ax1.set_ylabel(r'$||A{\theta}_{\epsilon} -b||_2^2$')
fig1.savefig("hw2_4_c_2_1.pdf")

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.plot(epsilon_list, norm_residual)
ax2.set_xlabel(r'${\epsilon}$')
ax2.set_ylabel(r'$||A{\theta}_{\epsilon} -b||_2^2$')
fig2.savefig("hw2_4_c_2_2.pdf")

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.plot(epsilon_list, norm_theta)
ax3.set_xlabel(r'${\epsilon}$')
ax3.set_ylabel(r'$||{\theta}_{\epsilon} ||_2^2$')
fig3.savefig("hw2_4_c_2_3.pdf")