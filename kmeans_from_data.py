print(__doc__)

import numbers
import numpy as np
from dataSet import *
from keras.datasets import mnist
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from sklearn.cluster import KMeans
from sklearn import datasets

np.random.seed(5)

centers = [[1, 1], [-1, -1], [1, -1]]

'''
iris = datasets.load_iris()
X = iris.data


input_file = "./data/iris.data.txt"
models = {} 

X_set = []
Y_set = []

data_out = open('data.txt', 'w')
for line in open(input_file, 'r'):
	data = line.split(',')[:-1]
	label = line.split(',')[-1]
	x = []
	for i in data:
		x.append(float(i))
	data_out.writelines(str(x).replace("[", "").replace("]", "").replace("'","").replace(",", '')+'\n')
	X_set.append(x)
	Y_set.append(int(label))
	
data_out.close()	
'''	

'''
input_file = "./data/processed.cleveland.data-out.txt"
models = {} 

X_set = []
Y_set = []

data_out = open('data.txt', 'w')
for line in open(input_file, 'r'):
	data = line.split(',')[:-1]
	label = line.split(',')[-1]
	x = []
	for i in data:
		x.append(float(i))
	X_set.append(x)
	data_out.writelines(str(x).replace("[", "").replace("]", "").replace("'","").replace(",", '')+'\n')
	l = int(label)
	if l==0:
		Y_set.append(0)
	else:
		Y_set.append(1)	


X_set = np.asarray(X_set)
data_out.close()
data = DataSet(X_set, Y_set)
data.representData()
X = data.X_set
'''

X_set = []
Y_set = []
for line in open('./data/bench_1_data.txt', 'r'):
	x = line.split('===')[0].replace('\n', '')
	y = line.split('===')[1].replace('\n', '')
	X_set.append(x.split())		
	Y_set.append(int(y))
	
data = DataSet(X_set, Y_set)
data.representData()

data_out = open('data.txt', 'w')
for x in data.X_set:
	xx = str(x).replace("[", "").replace("]", "").strip()
	xx = xx.split()
	xx = ' '.join(xx)
	data_out.writelines(xx+"\n")

data_out.close()

X = data.X_set

estimators = {#'k_means_iris_3': KMeans(n_clusters=3),
              #'k_means_iris_5': KMeans(n_clusters=5),
              #'k_means_iris_8': KMeans(n_clusters=8),              
              'k_means_iris_bad_init': KMeans(n_clusters=2, n_init=1,
                                              init='random')}



labels_out = open('labels.txt', 'w')
fignum = 1
for name, est in estimators.items():
    fig = plt.figure(fignum, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    plt.cla()
    est.fit(X)
    labels = est.labels_
#    print(labels)
    print(len(labels))    
    for x in labels:
    	labels_out.writelines(str(x)+'\n')

    ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=labels.astype(np.float))

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Petal width')
    ax.set_ylabel('Sepal length')
    ax.set_zlabel('Petal length')
    fignum = fignum + 1

# Plot the ground truth
fig = plt.figure(fignum, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
labels_out.close()


ax.scatter(X[:, 3], X[:, 0], X[:, 2])

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Petal width')
ax.set_ylabel('Sepal length')
ax.set_zlabel('Petal length')
plt.show()