import tensorflow as tf
import numpy as np

#生成数据
def createData(dataNum,w,b,sigma):
    x = np.arange(dataNum)
    # print(train_x)
    #根据x生成带噪声的y
    y = w*x+b+np.random.randn()*sigma
    return x,y


def linerRegression(train_x,train_y,epoch=30000,rate = 0.00001):
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    # n = train_x.shape[0]
    x = tf.placeholder("float")
    y = tf.placeholder("float")
    w = tf.Variable(tf.random_normal([1])) # 生成随机权重
    b = tf.Variable(tf.random_normal([1]))


    pred = tf.add(tf.multiply(x,w),b)
    #定义损失函数
    loss = tf.reduce_sum(tf.pow(pred-y,2))
    #使用梯度下降法,反向传播,训练数据
    optimizer = tf.train.GradientDescentOptimizer(rate).minimize(loss)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    print('weight start is ',sess.run(w))
    print('bias start is ',sess.run(b))
    for index in range(epoch):
        sess.run(optimizer,{x:train_x,y:train_y})
        if index%5000 == 0:
            print("by train {0} times the loss is {1}".format(index, sess.run(loss, {x:train_x, y:train_y})))
    print('final loss is ',sess.run(loss,{x:train_x,y:train_y}))
    w =  sess.run(w)
    b = sess.run(b)
    return w,b

def predictionTest(test_x,test_y,w,b):
    W = tf.placeholder(tf.float32)
    B = tf.placeholder(tf.float32)
    X = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)
    n = test_x.shape[0]
    pred = tf.add(tf.multiply(X,W),B)
    loss = tf.reduce_mean(tf.pow(pred-Y,2))
    sess = tf.Session()
    loss = sess.run(loss,{X:test_x,Y:test_y,W:w,B:b})
    return loss


if __name__ == "__main__":
    train_x,train_y = createData(50,2.0,7.0,0.5)
    test_x,test_y = createData(20,2.0,7.0,0.5)
    w,b = linerRegression(train_x,train_y)
    print('weights',w)
    print('bias',b)
    loss = predictionTest(test_x,test_y,w,b)
    print('loss',loss)
