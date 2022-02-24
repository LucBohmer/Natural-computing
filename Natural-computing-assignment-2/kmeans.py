import numpy as np
from sklearn.cluster import KMeans
from sklearn import datasets
import matplotlib.pyplot as plt
import random

#iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

#artificial dataset
def classification(vectors):
    ar_class = np.zeros(len(vectors[0]))

    for i in range(len(vectors[0])):
        if vectors[0][i] >= 0.7 or ((vectors[0][i] <= 0.3) and (vectors[1][i] >= -0.2 - vectors[0][i])):
            ar_class[i] = True
        else:
            ar_class[i] = False
    return ar_class

def generate_artificial_data():
    z1 = np.random.uniform(low=-1, high=1, size=400)
    z2 = np.random.uniform(low=-1, high=1, size=400)
    return np.array(list(zip(z1, z2, classification([z1, z2]))))

t_max = 10
# Generated problem
N_c = 2
data_truth = generate_artificial_data()
data_random_classification = []
for i in range(len(data_truth)):
    data_random_classification.append(np.append(data_truth[i][:-1],random.randint(0,N_c-1)))
data_random_classification = np.array(data_random_classification)
avg_acc = []

#Compute k-means clustering
for i in range(30):
    kmeans = KMeans(n_clusters=3,n_init=10,max_iter=100, init="random")
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)

    right_classification = 0
    for point_guess,point_truth in zip(y_kmeans,y):
        if point_guess == point_truth:
            right_classification+=1

    avg_acc.append(right_classification/len(y))
print(sum(avg_acc)/30)
fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(2, 1,1)
scatter1 = ax.scatter(X[:, 0],X[:, 1], c=y_kmeans)
ax.set_title("k_means plot")
ax.legend(handles=scatter1.legend_elements()[0], labels=['0', '1', '2'], title='classification')

ax = fig.add_subplot(2, 1, 2)
scatter2 = ax.scatter(X[:, 0],X[:, 1], c=y)
ax.set_title("true labels")
ax.legend(handles=scatter2.legend_elements()[0], labels=['0', '1', '2'], title='classification')

plt.show()