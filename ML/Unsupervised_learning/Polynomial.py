import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


import tensorflow as tf

learning_rate = 0.01
trainnig_epochs = 40
rng = np.random.RandomState(1)

def fun(x):
    a0,a1,a2,a3,e = 0.1,-0.02,0.03,-0.04,0.05
    y = a0 + a1 * x + a2 * (x**2) + a3 * (x**3)+ e
    y += 0.03 * rng.rand(1)
    return y

# 设置训练之前的值
trX = np.linspace(-1, 1, 30)
arrY = [fun(x) for x in trX]

#设置最高系数
num_coeffs = 4
trY = np.array(arrY).reshape(-1, 1)

X = tf.placeholder("float32")
Y = tf.placeholder("float32")

#设置模型
def model(X, w):
    terms = []
    for i in range(num_coeffs):
        term = tf.multiply(w[i], tf.pow(X, i))
        terms.append(term)
    return tf.add_n(terms)


w = tf.Variable([0.]*num_coeffs,  name="parameters")
print(w)
y_model = model(X, w)

cost = tf.reduce_sum(tf.square(Y-y_model))
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize((cost))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(trainnig_epochs):
        for (x, y) in zip(trX, trY):
            sess.run(train_op, feed_dict={X: x, Y: y})

    w_val = sess.run(w)
    print(w_val)


plt.figure()
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.title("Poly regression")

plt.scatter(trX, trY)
# plt.show()

trX2 = np.linspace(-1, 2, 100)
trY2 = 0
for i in range(num_coeffs):
    trY2 += w_val[i] * np.power(trX2, i)

plt.plot(trX2, trY2, 'ro')
plt.show()














"""
#使用sklearn解决
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

rng = np.random.RandomState(1)
#随机生成一个3x3的数组
print(rng.rand(3,3))

def fun(x):
    a0,a1,a2,a3,e = 0.1,-0.02,0.03,-0.04,0.05
    y = a0 + a1 * x + a2 * (x**2) + a3 * (x**3)+ e
    y += 0.03 * rng.rand(1)
    return y

plt.figure()
plt.title('polynomial regression(sklearn)')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)

X = np.linspace(-1, 1, 30)
arrY = [fun(x) for x in X]
# plt.plot(X, arrY, 'ro')

#为何要把它变成数组
X = X.reshape(-1, 1)
y = np.array(arrY).reshape(-1, 1)

plt.plot(X, y, 'k.')
# plt.show()

#建立模型
qf = PolynomialFeatures(degree=3)
qModel = LinearRegression()
qModel.fit(qf.fit_transform(X), y)

X_predict = np.linspace(-1, 2, 100)
X_predict_result = qModel.predict(qf.transform(X_predict.reshape(X_predict.shape[0], 1)))
plt.plot(X_predict, X_predict_result, 'b-')

plt.show()
"""
