# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 18:10:37 2020

@author: tbureete
"""

import csv
import cvxpy as cp
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

print('male training data set')
print(male_train_data[0:10])
print()
print('female training data set')
print(female_train_data[0:10])

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

b = np.concatenate((np.ones(male_train_data.shape[0]), 
                    -1*np.ones(female_train_data.shape[0])), axis=0)

theta = np.dot(np.linalg.inv(np.dot(A.T,A)),
               np.dot(A.T, b))

print()
print(theta)

x = cp.Variable(theta.shape[0])
objective = cp.Minimize(cp.sum_squares(A*x-b))
prob = cp.Problem(objective)
result = prob.solve()
print()
print(x.value)

fig=plt.figure()
ax=fig.add_subplot(111)
ax.scatter(male_train_data[:, 0], 
           male_train_data[:, 1], 
           color='r', 
           label='Male')
ax.scatter(female_train_data[:, 0], 
           female_train_data[:, 1], 
           color='b', 
           label='female')
ax.set_xlabel('BMI')
ax.set_ylabel('Stature (mm)')
ax.legend()
fig.savefig("hw2_3_a_1.pdf")

x1 = np.arange(np.amin(A[:, 0]), np.amax(A[:, 0]), 0.1)
x2 = -1*(theta[2]+(x1*theta[0]))/theta[1]

ax.plot(x1, x2, color='g')
fig.savefig("hw2_3_a_3.pdf")

male_test_data =[]
with open("male_test_data.csv", "r") as csv_file:
    reader = csv.reader(csv_file, delimiter=',')    
    for data in reader:
        male_test_data.append(data)        
    male_test_data.pop(0)   
csv_file.close()

female_test_data =[]
with open("female_test_data.csv", "r") as csv_file:    
    reader = csv.reader(csv_file, delimiter=',')    
    for data in reader:
        female_test_data.append(data)        
    female_test_data.pop(0)   
csv_file.close()

male_test_data = np.asarray(male_test_data, dtype=np.float32)
male_test_data = np.delete(male_test_data, 0, 1)
one_arr = np.resize(one_arr, (male_test_data.shape[0], 1))
male_test_data = np.concatenate((male_test_data, one_arr), axis=1)
male_test_result = np.dot(male_test_data, theta)
male_test_result = np.sign(male_test_result)
male_test_result = male_test_result + 1 
male_success = np.count_nonzero(male_test_result, axis=0)

female_test_data = np.asarray(female_test_data, dtype=np.float32)
female_test_data = np.delete(female_test_data, 0, 1)
one_arr = np.resize(one_arr, (female_test_data.shape[0], 1))
female_test_data = np.concatenate((female_test_data, one_arr), axis=1)
female_test_result = np.dot(female_test_data, theta)
female_test_result = np.sign(female_test_result)
female_test_result = female_test_result - 1
female_success = np.count_nonzero(female_test_result, axis=0)

success_rate = ((male_success + female_success)/
                (male_test_data.shape[0] + female_test_data.shape[0]))
print(success_rate)



