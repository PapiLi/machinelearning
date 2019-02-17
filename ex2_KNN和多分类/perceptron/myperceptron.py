import copy
import numpy as np
import pandas as pd
from sklearn import datasets
import os
import sys
import math
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score




'''
if y(wx+b) <=0, return false, else return true
'''

def shuffle_data(X, y, seed=None):
    if seed:
        #这个作用是每次产生一样的随机数
        np.random.seed(seed)
    #x.shape[0]返回的是行数，以前老师好像说过这里面其实每行是一条数据
    #np.arrange返回的是array([0,1,2,3,4])
    idx = np.arange(X.shape[0])
    #array内序号打乱
    np.random.shuffle(idx)
    return X[idx], y[idx]


# 划分数据集为训练集和测试集
def train_test_split(X, y, test_size=0.2, shuffle=True, seed=None):
    if shuffle:
        X, y = shuffle_data(X, y, seed)

    n_train_samples = int(X.shape[0] * (1 - test_size))
    x_train, x_test = X[:n_train_samples], X[n_train_samples:]
    y_train, y_test = y[:n_train_samples], y[n_train_samples:]

    return x_train, x_test, y_train, y_test


data = pd.read_csv('adult_for.data')
train_np = data.values
# print(train_np)
X_train, X_test, y_train, y_test = train_test_split(train_np[:, 1:], train_np[:, 0], test_size=0.33, shuffle=True)
w = np.ones(shape=(X_train.shape[1], 1))
b = 0
yita = 0.5
record = []

def sign(x,y):
    global w, b
    res = 0
    x = np.array(x)
    x = x.reshape([42, 1])
    res = sum(x * w + b)*y
    if res >0: return 1
    else: return -1

def update(x,y):
    global w, b, record
    X = x.reshape([42, 1])
    w = w + yita * y * X

    # print('2', type(w), w.shape)
    b = b + yita * y

    record.append([copy.copy(w), b])

def perceptron():
    count = 1
    for x, y in zip(X_train, y_train):
        # print(x, type(x))
        flag = sign(x, y)
        if not flag > 0:
            # count = 1
            update(x, y)
        else:
            count += 1
        if count >= len(X_train):
            return 1
    return count

def predict(x):
    global w, b
    x = np.array(x)
    x = x.reshape([42, 1])
    res = sum(x * w + b)
    if res >0: return 1
    else: return -1

def accurary(y, y_pred):
    y = y.reshape(y.shape[0], -1)
    y_pred = np.array(y_pred)
    y_pred = y_pred.reshape(y_pred.shape[0], -1)
    return np.sum(y==y_pred)/len(y)

"""
x1 = []
y1 = []
x2 = []
y2 = []

fig = plt.figure()
ax = plt.axes(xlim = (-1, 5), ylim = (-1, 5))
line, = ax.plot([], [], 'g', lw=2)

def init():
    line.set_data([], [])
    for p in data:
        if p[1] > 0:
            x1.append(p[0][0])
            y1.append(p[0][1])
        else:
            x2.append(p[0][0])
            y2.append(p[0][1])
    plt.plot(x1, y1, 'or')
    plt.plot(x2, y2, 'ob')
    # print('x1 y1 x2 y2: ', x1, y1, x2, y2)
    return line,

def animate(i):
    global record, ax, line
    w = record[i][0]
    b = record[i][1]
    x1 = -5
    y1 = -(b+w[0]*x1)/w[1]
    x2 = 6
    y2 = -(b+w[0]*x2)/w[1]
    line.set_data([x1, x2], [y1, y2])
    print(type(line), line)
    print([x1, x2], [y1, y2])
    return line,
"""

if __name__ == '__main__':

    perceptron()

    y_pred = []
    for x in X_test:
        y_pred.append(predict(x))
    accu = accurary(y_test, y_pred)
    print("accurary: ", accu)








