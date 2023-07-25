# -*- coding: utf-8 -*-

import csv
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

train_data =[]
with open("hw04_sample_vectors.csv", "r") as csv_file:
    reader = csv.reader(csv_file, delimiter=',')    
    for data in reader:
        train_data.append(data)        
csv_file.close()

label =[]
with open("hw04_labels.csv", "r") as csv_file:    
    reader = csv.reader(csv_file, delimiter=',')    
    for data in reader:
        label.append(data)        
csv_file.close()

for data in label:
    if np.linalg.norm(data) == 0:
        data[0] = -1
        
x2 = np.arange(-0.6, 0.4, 0.01)
        
def plot_graph(name, omega, omega_0):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(train_data[:1000, 0], train_data[:1000, 1], c='b', s=3)
    ax1.scatter(train_data[1000:, 0], train_data[1000:, 1], c='r', s=3)
    x1 = (-1*omega[1]*x2 - omega_0) / omega[0]
    ax1.plot(x1, x2, 'g-')
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    fig.savefig('hw4_2_c_{}.pdf'.format(name))
        
label = np.asarray(label, dtype=np.float32)
train_data = np.asarray(train_data, dtype=np.float32)
#one_arr = np.ones(train_data.shape[0])
#one_arr = np.resize(one_arr, (train_data.shape[0], 1))
#train_data = np.concatenate((train_data, one_arr), axis=1)

omega = cp.Variable(2)
omega_0 = cp.Variable(1)
xi = cp.Variable(2000)
C = 1

objective = cp.Minimize(cp.norm(omega)/2 + C*cp.sum_squares(xi))
constraint1 = [label[i] * (omega.T * train_data[i] + omega_0) >= 1-xi[i] 
                for i in range(len(label))] 
constraint2 = [xi[i] >= 0 for i in range(len(label))]
constraints = constraint1 + constraint2
prob = cp.Problem(objective, constraints)

result = prob.solve()

print(omega.value)
print(omega_0.value)

plot_graph('2', omega.value, omega_0.value)