# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 00:13:15 2020

@author: tbureete

"""

import numpy as np
import matplotlib.pyplot as plt

def compute_mean(sample):
    dimension = sample.shape[0]
    sum_vector = np.zeros((dimension,1))
    sum_vector = sum_vector.flatten('F') 
    
    for x in range(sample.shape[1]):
        sum_vector = sum_vector + sample[:,x].flatten('F')

    return (sum_vector/sample.shape[1]).T

def compute_cov(sample):
    mean = compute_mean(sample)
    dimension = sample.shape[0]
    err_sum_vector = np.zeros((dimension, dimension))
    
    for x in range(sample.shape[1]):
        err_vector = sample[:,x].flatten('F').T - mean
        err_sum_vector = err_sum_vector + np.dot(err_vector, err_vector.T)
        
    return (err_sum_vector/(sample.shape[1]-1))

def compute_prior(samples):
    total_sample = 0
    for sample in samples:
        total_sample = total_sample + sample.shape[1]
        
    prior = []
    for sample in samples:
        prior.append(sample.shape[1]/total_sample)
        
    return prior

train_cat = np.matrix(np.loadtxt('train_cat.txt', delimiter = ','))
train_grass = np.matrix(np.loadtxt('train_grass.txt', delimiter = ','))

cat_mean = compute_mean(train_cat)
grass_mean = compute_mean(train_grass)
cat_cov = compute_cov(train_cat)
grass_cov = compute_cov(train_grass)
prior_prob = compute_prior([train_grass, train_cat])
d = train_cat.shape[0]

params = [{
    'class': 'grass',
    'value': 0,
    'mean': grass_mean,
    'cov': grass_cov,
    'inv_cov': np.linalg.inv(grass_cov),
    'det_cov': np.linalg.det(grass_cov),
    'log_det_cov': (1/2) * np.log(np.linalg.det(grass_cov)),
    'prior_prob': prior_prob[0],
    'log_prior_prob': np.log(prior_prob[0])
    }, {
    'class': 'cat',
    'value': 1,
    'mean': cat_mean,
    'cov': cat_cov,
    'inv_cov': np.linalg.inv(cat_cov),
    'det_cov': np.linalg.det(cat_cov),
    'log_det_cov': (1/2) * np.log(np.linalg.det(cat_cov)),
    'prior_prob': prior_prob[1],
    'log_prior_prob': np.log(prior_prob[1])
    }]

Y = plt.imread('cat_grass.jpg') / 255
output = np.zeros((Y.shape[0]-8, Y.shape[1]-8))
pi_term = (d/2) *  np.log(2*np.pi)
for i in range(Y.shape[0]-8):
    for j in range(Y.shape[1]-8):
        x = Y[i:i+8, j:j+8]
        x = x.flatten('F')
        g = []
        for param in params:
            diff = (x - param['mean'].flatten('F')).reshape((d, 1))
            g.append((param['log_prior_prob']
                     - pi_term
                     - param['log_det_cov']
                      - (1/2) * np.dot(diff.T, 
                         np.dot(param['inv_cov'], diff)))[0 ,0])       
                    
        output[i, j] = params[g.index(max(g))]['value']
        
plt.imshow(output*255, cmap = 'gray')
plt.savefig("hw3_2_b_1.pdf")
            
output2 = np.zeros((Y.shape[0]-8, Y.shape[1]-8))
for i in range(Y.shape[0]-8):
    if (i % 8) != 0 or i+8 >= Y.shape[0]-8:
        continue
    for j in range(Y.shape[1]-8):
        if (j % 8) != 0 or j+8 >= Y.shape[1]-8:
            continue
        x = Y[i:i+8, j:j+8]
        x = x.flatten('F')
        g = []
        for param in params:
            diff = (x - param['mean'].flatten('F')).reshape((d, 1))
            g.append((param['log_prior_prob']
                     - pi_term
                     - param['log_det_cov']
                      - (1/2) * np.dot(diff.T, 
                         np.dot(param['inv_cov'], diff)))[0 ,0])       
        
        value = params[g.index(max(g))]['value']
        for m in range(8):
            for n in range(8):
                try:
                    output2[i+m, j+n] = value
                except IndexError:
                    print(f'({i}+{m}, {j}+{n})')
        
plt.imshow(output2*255, cmap = 'gray')  
plt.savefig("hw3_2_b_2.pdf")    

GT = plt.imread('truth.png')          

MAE1 = np.sum(np.absolute(GT[:output.shape[0],:output.shape[1]]-output))/output.size
MAE2 = np.sum(np.absolute(GT[:output2.shape[0],:output2.shape[1]]-output2))/output2.size

print(MAE1)
print(MAE2)
                
                 
                
            
            
            
            
            
            
            
            
            
            
