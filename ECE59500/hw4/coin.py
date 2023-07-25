# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

coins = np.random.binomial(10, .5, 1000)

np.random.seed = 1
rand = np.random.randint(0, 1000, 1)

print(coins[0]/10)
print(coins[rand][0]/10)
print(np.min(coins)/10)

coins = np.resize(coins, (1000, 1))
for i in range(100000-1):
    coin = np.random.binomial(10, .5, 1000)
    coin = np.resize(coin, (1000, 1))
    coins = np.concatenate((coins, coin), axis=1)
    
min_index = np.where(coins == np.min(coins, axis=0))[0][0]

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.hist(coins[0], bins=11)
fig1.savefig('hw4_3_b_1.pdf')

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.hist(coins[rand[0]], bins=11)
fig2.savefig('hw4_3_b_2.pdf')

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.hist(coins[min_index], bins=11)
fig3.savefig('hw4_3_b_3.pdf')

index = [1, rand[0], min_index]
coins = coins/10
epsilon_list = np.arange(0,0.51, 0.05)
p_array = np.zeros((3, 1))
h_bounds = []

for epsilon in epsilon_list:
    h_bound = 2*np.exp(-2*(epsilon**2)*coins.shape[1])
    h_bounds.append(h_bound)
    p_list = []
    for i in index:
        p = 0
        for j in range(coins.shape[1]):
            if abs(coins[i][j] - 0.5) > epsilon:
                p = p+1
        p_list.append(p)
    p_array = np.concatenate((p_array, np.reshape(np.asarray(p_list), (3, 1))), axis=1)
    
p_array = np.delete(p_array, 0, axis=1)
p_array = p_array/coins.shape[1]

fig4 = plt.figure()
ax4 = fig4.add_subplot(111)
ax4.plot(epsilon_list, h_bounds, 'black')
ax4.scatter(epsilon_list, p_array[0], c='r', s=10, label='$c_1$')
ax4.scatter(epsilon_list, p_array[1], c='g', s=10, label='$c_{rand}$')
ax4.scatter(epsilon_list, p_array[2], c='b', s=10, label='$c_{min}$')
ax4.legend()
fig4.savefig('hw4_3_c_1.pdf')
