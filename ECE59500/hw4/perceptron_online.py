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
        
def plot_graph(name, theta):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(train_data[:1000, 0], train_data[:1000, 1], c='b', s=3)
    ax1.scatter(train_data[1000:, 0], train_data[1000:, 1], c='r', s=3)
    x1 = ( -1*theta[1]*x2 - theta[2]) / theta[0]
    ax1.plot(x1, x2, 'g-')
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    fig.savefig('hw4_2_b_{}.pdf'.format(name))
        
label = np.asarray(label, dtype=np.float32)
train_data = np.asarray(train_data, dtype=np.float32)
one_arr = np.ones(train_data.shape[0])
one_arr = np.resize(one_arr, (train_data.shape[0], 1))
train_data = np.concatenate((train_data, one_arr), axis=1)

iteration = 100
theta = np.array([1 , 1, 1])
theta_list = []

for k in range(iteration):
    theta_list.append(theta)
    shuffled_index = np.random.permutation(label.size) 
    x = train_data[shuffled_index, :] 
    y = label[shuffled_index] 
    for i in range(len(y)):
        if np.dot(theta, x[i])*y[i] < 0:
            theta = theta + 0.1*y[i]*x[i]
            break
    if i == 1999:
        break

checkpoint = [k*0.2, k*0.4, k*0.6, k*0.8]
for point in checkpoint:
    plot_graph(int(point), theta_list[int(point)])
        
plot_graph(k+1, theta)

loss = 0    
for i in range(len(label)):
    loss = loss + max(0, -1*np.dot(theta, train_data[i])*label[i])
print(loss)