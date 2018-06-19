"""
实现了k-means聚合
使用了datasets pd.DataFrame()生成所需要的数据
标准化
建立KMeans建立模型
fit训练模型
labels_显示类别数
cluster_centers_ 表示聚类中心
sklearn.decomposition import PCA 进行降维
plt.plot()进行图形的可视化

"""

import pandas as pd
from sklearn import datasets

##需要聚类的样本150个 4个变量
#初始化150x4的数组
iris = datasets.load_iris()
print(type(iris.data), len(iris.data))
# 把它变成类似csv格式的
data = pd.DataFrame(iris.data)
print(dir(data))
print(type(data))

#数据标准化(z-score)
data_zs = (data-data.mean())/data.std()
print(data_zs)

#导入sklearn中的kmeans
from sklearn.cluster import KMeans

#设置类数K
k = 3

#设置最大迭代数
iteration = 1000

"""
（1）对于K均值聚类，我们需要给定类别的个数n_cluster，默认值为8； 
（2）max_iter为迭代的次数，这里设置最大迭代次数1000； 
（3）n_init设为10意味着进行10次随机初始化，选择效果最好的一种来作为模型； 
（4） init=’k-means++’ 会由程序自动寻找合适的n_clusters； 
（5）tol：float形，默认值= 1e-4，与inertia结合来确定收敛条件； 
（6）n_jobs：指定计算所用的进程数； 
（7）verbose 参数设定打印求解过程的程度，值越大，细节打印越多； 
（8）copy_x：布尔型，默认值=True。当我们precomputing distances时，将数据中心化会得到更准确的结果。如果把此参数值设为True，则原始数据不会被改变。如果是False，则会直接在原始数据 
上做修改并在函数返回值时将其还原。但是在计算过程中由于有对数据均值的加减运算，所以数据返回后，原始数据和计算前可能会有细小差别。
"""
#设置kmeans的对象
model = KMeans(n_clusters=k, n_jobs=4, max_iter=iteration)

#使用数据训练models
model.fit(data_zs)

#每个类别样本个数
count = pd.Series(model.labels_).value_counts()
print(type(count))
print("每个样本的个数: \n",count)

#每个类别的聚类中心
cluster_center = pd.DataFrame(model.cluster_centers_)
print(cluster_center)

"""用高维可视化工具TSNE对聚类结果进行可视化"""
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

tsne = TSNE(learning_rate=100)

#对数据进行降维
tsne.fit_transform(data_zs)
data = pd.DataFrame(tsne.embedding_, index=data_zs.index)

#使用不同类别用不同颜色和样式绘制图形
# d = data[model.labels_==0]
# plt.plot(d[0], d[1], 'r.')
# d = data[model.labels_==1]
# plt.plot(d[0], d[1], 'go')
# d = data[model.labels_==2]
# plt.plot(d[0], d[1], 'b*')

# plt.show()

"""用PCA降维后 对聚类结果进行可视化"""
from sklearn.decomposition import PCA
pca = PCA()
print("降维之前的效果\n", data_zs)
data = pca.fit_transform(data_zs)
print("降维之后的效果\n",data)
data = pd.DataFrame(data, data_zs.index)

#这里k等于多少就设置多少个model.labels
d = data[model.labels_==0]
plt.plot(d[0], d[1], 'r.')
d = data[model.labels_==1]
plt.plot(d[0], d[1], 'go')
d = data[model.labels_==2]
plt.plot(d[0], d[1], 'b*')
# d = data[model.labels_==3]
# plt.plot(d[0], d[1], 'yv')


plt.show()



