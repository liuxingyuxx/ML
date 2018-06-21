import numpy as np
from numpy.linalg import cholesky
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
from random import shuffle
from numpy import array


############基于tensorflow###############
def KMeansCluster(vectors, noofclusters):
    """
    vertors`是一个n*k的二维的NumPy的数组，其中n代表着K维向量的数目------总的待分类数目
    """
    #要划分簇的数目
    noofclusters = int(noofclusters)

    #划分簇的数目必须小于节点个数
    assert noofclusters < len(vectors)

    #找出每个向量的维度
    dim = len(vectors[0])

    #辅助----随机地从可得的向量中选取中心点
    vector_indices = list(range(len(vectors)))
    # shuffle(vector_indices)

    # print("shuffle:\n", len(vectors),shuffle(vector_indices))

    #计算图
    #我们创建了一个默认的计算流的图用于整个算法中，这样就保证了当函数被多次调用
    #时，默认的图并不会被从上一次调用时留下的未使用的OPS或者Variables挤满
    graph = tf.Graph()

    with graph.as_default():
        sess = tf.Session()

        ##构建基本的计算的元素
        ##首先我们需要保证每个中心点都会存在一个Variable矩阵
        ##从现有的点集合中抽取出一部分作为默认的中心点
        centroids = [tf.Variable((vectors[vector_indices[i]]))
                     for i in range(noofclusters)]

        ##创建一个placeholder用于存放各个中心点可能的分类的情况----每个中心点存储在cent_assigns中
        centroid_value = tf.placeholder("float64", [dim])
        cent_assigns = []

        for centroid in centroids:
            cent_assigns.append(tf.assign(centroid, centroid_value))

        # print("中心点:" , cent_assigns)
        """目标"""
        ##对于每个独立向量的分属的类别设置为默认值0-----------------设置默认为第0簇
        assignments = [tf.Variable(0) for i in range(len(vectors))]

        ##这些节点在后续的操作中会被分配到合适的值
        assignment_value = tf.placeholder("int32")


        cluster_assigns = []

        for assignment in assignments:
            cluster_assigns.append(tf.assign(assignment,
                                             assignment_value))


        ##下面创建用于计算平均值的操作节点
        #输入的placeholder
        mean_input = tf.placeholder("float", [None, dim])

        """目标2 调整每个簇的中心点"""
        #节点/OP接受输入，并且计算0维度的平均值，譬如输入的向量列表
        #指定第二个参数为0，则第一维的元素取平均值，即每一列求平均值
        mean_op = tf.reduce_mean(mean_input, 0)

        ##用于计算欧几里得距离的节点----reduce_sum
        v1 = tf.placeholder("float", [dim])
        v2 = tf.placeholder("float", [dim])
        euclid_dist = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(v1, v2), 2)))

        """目标"""
        ##这个OP会决定应该将向量归属到哪个节点
        ##基于向量到中心点的欧几里得距离---------------维度为簇的个数即每个向量到各个簇的距离
        centroid_distances = tf.placeholder("float", [noofclusters])
        #寻找最小距离的簇
        cluster_assignment = tf.argmin(centroid_distances, 0)
        # print("到中心节点的距离:", tf.Session().run(centroid_distances))

        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        ##集群遍历
        #接下来在K-Means聚类迭代中使用最大期望算法。为了简单起见，只让它执行固定的次数，而不设置一个终止条件
        noofiterations = 100

        for iteration_n in range(noofiterations):

            ##期望步骤
            ##基于上次迭代后算出的中心点的未知
            ##the _expected_ centroid assignments.

            #首先遍历所有的向量
            for vector_n in range(len(vectors)):
                vect = vectors[vector_n]

                #计算给定向量与分配的中心节点之间的欧几里得距离 用数组存储与每个中心节点的距离
                distances = [sess.run(euclid_dist, feed_dict={
                    v1: vect, v2: sess.run(centroid)})
                             for centroid in centroids]

                #下面可以使用集群分配操作，将上述的距离当做输入
                assignment = sess.run(cluster_assignment, feed_dict = {
                    centroid_distances: distances})


                #接下来为每个向量分配合适的值 即分配到距离最短的中心簇
                sess.run(cluster_assigns[vector_n], feed_dict={
                    assignment_value: assignment})

            ##最大化的步骤
            """目标2 调整每个簇的中心点"""
            #基于上述的期望步骤，计算每个新的中心点的距离从而使集群内的平方和最小
            for cluster_n in range(noofclusters):
                #收集所有分配给该集群的向量
                assigned_vects = [vectors[i] for i in range(len(vectors))
                                  if sess.run(assignments[i]) == cluster_n]
                #计算新的集群中心点
                new_location = sess.run(mean_op, feed_dict={
                    mean_input: array(assigned_vects)})
                #为每个向量分配合适的中心点
                sess.run(cent_assigns[cluster_n], feed_dict={
                    centroid_value: new_location})

        #返回中心节点和分组
        centroids = sess.run(centroids)
        assignments = sess.run(assignments)
        return centroids, assignments



############生成测试数据###############

sampleNo = 100                  #数据数量


# 二维正态分布
mu = np.array([[1, 5]])
Sigma = np.array([[1, 0.5], [1.5, 3]])
# print(Sigma)
R = cholesky(Sigma)
print(R)
srcdata= np.dot(np.random.randn(sampleNo, 2), R) + mu
plt.plot(srcdata[:,0],srcdata[:,1],'bo')




############kmeans算法计算###############
k=3
center,result=KMeansCluster(srcdata,k)
print("每个簇的中心节点为:",center)





############利用seaborn画图###############

res={"x":[],"y":[],"kmeans_res":[]}
for i in range(len(result)):
    res["x"].append(srcdata[i][0])
    res["y"].append(srcdata[i][1])
    res["kmeans_res"].append(result[i])
print(res['kmeans_res'])
#创建DataFrame对象的数据 添加了行索引和列索引
pd_res=pd.DataFrame(res)

print(pd_res)

#fig_reg代表不回归  size明确图的大小
sns.lmplot("x","y",data=pd_res,fit_reg=False,size=8,hue="kmeans_res")
plt.show()
