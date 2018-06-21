import tensorflow as tf
import numpy as np

'''
batch_size
    input_size---输入层的大小
    state_size---隐藏层的大小
    output_size---输出层的大小
'''

lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=128) #state_size=128

inputs = tf.placeholder(np.float32, shape=(32, 100)) #32是batch_size
h0 = lstm_cell.zero_state(32, np.float32)
# output, h1 = lstm_cell.call(inputs, h0)

# print(h1.h)
