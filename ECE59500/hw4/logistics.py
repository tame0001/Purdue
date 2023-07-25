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

label = np.asarray(label, dtype=np.float32)
train_data = np.asarray(train_data, dtype=np.float32)
one_arr = np.ones(train_data.shape[0])
one_arr = np.resize(one_arr, (train_data.shape[0], 1))
train_data = np.concatenate((train_data, one_arr), axis=1)

x2 = np.arange(-0.6, 0.4, 0.01)

def plot_graph(name, theta):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(train_data[:1000, 0], train_data[:1000, 1], c='b', s=3)
    ax1.scatter(train_data[1000:, 0], train_data[1000:, 1], c='r', s=3)
    x1 = (0.5 - theta[1]*x2 - theta[2]) / theta[0]
    ax1.plot(x1, x2, 'g-')
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    fig.savefig('hw4_2_a_{}.pdf'.format(name))

iteration = 100
checkpoint = [iteration*0.2, iteration*0.4, iteration*0.6, iteration*0.8]
learning_rate = 0.1
theta = np.array([1 , 1, 1])
norms = []

for k in range(iteration):
    cost = np.zeros(3)
    for i in range(len(label)):
        h = 1 / (1 + np.exp(-1*np.dot(train_data[i], theta)))
        cost = cost + ((h-label[i]) * train_data[i])
    
    theta = theta - learning_rate * cost
    norms.append(np.linalg.norm(theta))
    
    if k in checkpoint:
        plot_graph(k, theta)

plot_graph(k, theta)

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.plot(list(range(1, 101)), norms, '-')
ax2.set_ylabel(r'$||\theta||_2$')
ax2.set_xlabel('iteration')
fig2.savefig('hw4_2_a_1.pdf')

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.plot(list(range(1, 101))[-20:], norms[-20:], '-')
ax3.set_ylabel(r'$||\theta||_2$')
ax3.set_xlabel('iteration')
fig3.savefig('hw4_2_a_2.pdf')

