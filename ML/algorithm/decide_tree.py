from sklearn import tree
# 导入sklearn自带的数据集
from sklearn.datasets import load_iris
from sklearn.externals.six import StringIO
import pydotplus
from collections import defaultdict

# Data Collection
X = [ [180, 15,0],
      [177, 42,0],
      [136, 35,1],
      [174, 65,0],
      [141, 28,1]]

Y = ['man', 'woman', 'woman', 'man', 'woman']

data_feature_names = [ 'height', 'hair length', 'voice pitch' ]


# Training
clf = tree.DecisionTreeClassifier(splitter='best')
clf = clf.fit(X,Y)


# Visualize data
dot_data = tree.export_graphviz(clf,
                                feature_names=data_feature_names,
                                out_file=None,
                                filled=True,
                                rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)

colors = ('turquoise', 'orange')
edges = defaultdict(list)

for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))

for edge in edges:
    edges[edge].sort()
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])

graph.write_png('tree.png')








    # :Attribute Information:
    #     - sepal length in cm   花萼
    #     - sepal width in cm
    #     - petal length in cm
    #     - petal width in cm
    #     - class:
    #             - Iris-Setosa
    #             - Iris-Versicolour
    #             - Iris-Virginica
iris = load_iris()
print(iris.keys())
# print(iris.data)
print(iris.DESCR)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)


dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")


