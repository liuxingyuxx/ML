"""
逻辑回归 自我实现 不依赖已有模型
使用最大似然估计的方法
实现细节方面还不是很清楚
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
从一个均匀分布[low,high)中随机采样 
size=(m,n,k) 缺省时为1 输出的数组
"""
data = np.random.uniform(low=-5, high=5, size=(100, 2))
print(data)
data = pd.DataFrame(data)
print(data)

#iloc=index location
print(data.iloc[:, 0])
#生成true or false
mask = (data.iloc[:, 0] + 0.5* data.iloc[:, 1])<0
print(mask)

#将true or false 改成 0 或者 1
data['y']=mask*1
print(data['y'])
print(data.iloc[:, 2])
print(data['y']==data.iloc[:, 2])#两者相同

data1 = data[data.iloc[:, 2] == 1] #为了画图，两类不同颜色
data2 = data[data.iloc[:, 2] == 0]
plt.plot(data1.iloc[:, 0], data1.iloc[:, 1], 'ro')
plt.plot(data2.iloc[:, 0], data2.iloc[:, 1], 'b*')
# plt.show()

"""迭代求解"""
alpha = 0.001 #步长
step = 500
m, n = data.shape #100x3
weights = np.ones((n, 1))  #此时形状为3x1 全为1 的矩阵


#去掉y列 并添加全1的第一列
data_x = np.concatenate((np.ones((m, 1)), np.array(data.iloc[:, :2])), axis=1)
print(data_x)
target = np.array(data.iloc[:, 2])
target.shape = -1, 1  #将一个数组变成中的每个元素都变成一个数组(1x1)

#这是逻辑回归的基本算法
def logistic(wTx):
    return 1 / (1 + np.exp(-wTx))


for i in range(step):
    wTx = np.dot(data_x, weights)  #dot表示矩阵积 即点积
    output = logistic(wTx)
    errors = target - output
    #numpy.ndarray.T是转置!
    weights = weights + alpha*np.dot(data_x.T, errors)

X = np.linspace(-5, 5, 100) #随机生成范围从-5到5的一百个数 组成一个list 作为初始值
print("weights[0],weights[1],weights[2]分别为{0},{1},{2}".format(weights[0], weights[1], weights[2]))
Y = -(weights[0] + X * weights[1]) / weights[2]

plt.plot(X,Y)
# plt.plot(data1.iloc[: 0], data1.iloc[:, 1], 'ro')
# plt.plot(data2.iloc[: 0], data2.iloc[:, 1], 'b^')
plt.show()



