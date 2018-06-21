import numpy as np
import tensorflow as tf
from tensorflow.contrib import model_pruning
import matplotlib.pyplot as plt
from tensorflow.contrib.learn.python.learn.estimators.estimator import SKCompat
from matplotlib import style
import pandas as pd


"""数据预处理"""

#####读入原始数据并转为list####
path = '/home/liuxinyu/download/'
data = pd.read_csv(path+'AirPassengers.csv')
data_x = data.iloc[:,1].tolist()
data_y = data.iloc[:,0].tolist()

'''自定义数据尺度缩放函数'''
def data_processing(raw_data,scale=True):
    if scale == True:
        return (raw_data-np.mean(raw_data))/np.std(raw_data)#标准化
    else:
        return (raw_data-np.min(raw_data))/(np.max(raw_data)-np.min(raw_data))#极差规格化

'''观察数据'''

'''设置绘图风格'''
style.use('ggplot')
print(len(data))
print(data)
# data = data_processing(data)
plt.plot(data_x, data_y)
plt.show()


'''
这里我们需要设置的参数有隐层层数，因为数据集比较简单，我们设置为1；隐层神经元个
数，这里我随意设置为40个；时间步中递归次数，这里根据上面观察的结论，设置为12；
训练轮数，这里也是随意设置的不宜过少，2000；训练批尺寸，这里随意设置为20，
表示每一轮从训练集中抽出20组序列样本进行训练：
'''
HiDDLE_SIZE = 40
NUM_LAYERS = 1
TIMESTEPS = 12
TRAINING_STEPS = 2000
BATCH_SIZE = 20


'''
　　这里为了将原始的单变量时序数据处理成LSTM可以接受的数据类型（有X输入，有
真实标签Y），我们通过自编函数，将原数据（144个）从第一个开始，依次采样长度为12的
连续序列作为一个时间步内部的输入序列X，并采样其之后一期的数据作为一个Y
'''

#样本生成函数
def generate_data(seq):
    X = []
    Y = []
