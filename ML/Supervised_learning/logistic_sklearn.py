"""
这是基于sklearn已有模型实现的逻辑回归分类方法
"""

"""step1 建立模型"""

#随机生成一个字典格式 dict_keys(['target_names', 'data', 'feature_names', 'DESCR', 'target'])
#是其自带的鸢尾花的数据集
from sklearn.datasets import  load_iris

dataset = load_iris()  #data长度为150x4的矩阵
print([dataset.keys()])
print(dataset.get('target'))
#print(dataset['data'])

from sklearn import linear_model
clf = linear_model.LogisticRegression()
clf.fit(dataset.data, dataset.target)


"""模型的使用"""
cls = clf.predict(dataset.data) #判读数据属于哪个类别
print(cls)
cls_proba = clf.predict_proba(dataset.data) #判断属于各个类别的准确率
print(cls_proba)
proba = clf.score(dataset.data, dataset.target)

"""
clf.coef_ 系数
clf.intercept_ 截距
clf.n_iter_ 迭代系数
"""



