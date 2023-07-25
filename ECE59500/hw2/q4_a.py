# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 12:21:25 2020

@author: tbureete
"""

import csv
import numpy as np
import matplotlib.pyplot as plt

male_train_data =[]
with open("male_train_data.csv", "r") as csv_file:
    reader = csv.reader(csv_file, delimiter=',')    
    for data in reader:
        male_train_data.append(data)        
    male_train_data.pop(0)   
csv_file.close()

female_train_data =[]
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

norm_theta_lambda = []
norm_residual_lambda = []
theta_list = []
lambda_list = np.arange(0.1, 10, 0.1)
for l in lambda_list:
    theta_lambda = np.dot(np.linalg.inv(np.dot(A.T,A) + 
                                        l*np.eye(A.shape[1])), 
                          np.dot(A.T, b))
    theta_list.append(theta_lambda)
    norm_theta_lambda.append(np.linalg.norm(theta_lambda)**2)
    residual = np.dot(A, theta_lambda) - b
    norm_residual_lambda.append(np.linalg.norm(residual)**2)
    
norm_theta_lambda = np.asarray(norm_theta_lambda)
norm_residual_lambda = np.asarray(norm_residual_lambda)
theta_list = np.asarray(theta_list)

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(norm_theta_lambda, norm_residual_lambda)
ax1.set_xlabel(r'$||{\theta}_\lambda ||_2^2$')
ax1.set_ylabel(r'$||A{\theta}_\lambda -b||_2^2$')
fig1.savefig("hw2_4_a_1.pdf")

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.plot(lambda_list, norm_residual_lambda)
ax2.set_xlabel(r'$\lambda$')
ax2.set_ylabel(r'$||A{\theta}_\lambda -b||_2^2$')
fig2.savefig("hw2_4_a_2.pdf")

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.plot(lambda_list, norm_theta_lambda)
ax3.set_xlabel(r'$\lambda$')
ax3.set_ylabel(r'$||{\theta}_\lambda ||_2^2$')
fig3.savefig("hw2_4_a_3.pdf")

x = np.linspace(np.amin(A[:, 0]), np.amax(A[:, 0]), 200)
legend_str = []
plt.figure(figsize=(15,7.5))
for i in range(len(lambda_list))[0::10]:
    y = -1*(theta_list[i][2]+(x*theta_list[i][0]))/theta_list[i][1]  
    plt.plot(x, y.T)
    legend_str.append('$\lambda = $' + str(lambda_list[i]))
plt.legend(legend_str)
plt.xlabel('BMI')
plt.ylabel('Stature (dm)')
plt.savefig("hw2_4_a_4.pdf")