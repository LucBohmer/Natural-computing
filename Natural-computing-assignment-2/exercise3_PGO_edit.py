import numpy as np
import matplotlib.pyplot as plt
import random
import sys
from sklearn import datasets
np.set_printoptions(threshold=sys.maxsize)


def euclidean_distance(point, centroid):
    return np.linalg.norm(centroid[:-1] - point[:-1])

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

def generate_centroids(data, N_c, N_p):
    particle_centroids = []

    for n in range(N_p):
        particle_centroids.append(random.sample(list(np.copy(data)), N_c))
    return particle_centroids


def fitness(data, particle, N_c):
    nom = 0

    for j in range(N_c):
        data_j = data[data[:,N_c] == j]

        for Z_p in data_j:
            nom += euclidean_distance(Z_p, particle[j]) / len(data_j)

    return (nom / N_c)

def update_velocity(w, v, x, x_hat, g_hat):
    return w * v + np.dot((a_1 * r_1) ,(x_hat - x)) + np.dot((a_2 * r_2), (g_hat - x))
    

t_max = 100
trails = 30

# Generated problem
# N_c = 2
# Acc = 2
# data_truth = generate_artificial_data()
# data_random_classification = []
# for i in range(len(data_truth)):
#     data_random_classification.append(np.append(data_truth[i][:-1],random.randint(0,N_c-1)))
# data_random_classification = np.array(data_random_classification)
# print(data_truth)


#Iris
iris=datasets.load_iris()
X = iris.data.tolist()
y = iris.target.tolist()

N_c = 3
Acc = 4
data_truth = []
data_random_classification = []
for i in range(len(X)):
    data_truth.append([X[i][0],X[i][1], X[i][2], X[i][3], y[i]])
    data_random_classification.append([X[i][0],X[i][1], X[i][2], X[i][3], random.randint(0,2)])
data_truth = np.array(data_truth)
data_random_classification = np.array(data_random_classification)
# print(data_truth)


N_p = 10 
v = 0
w = 0.72
a_1, a_2 = 1.49, 1.49
r_1, r_2 = random.random(), random.random()

accuracies = []
for trail in range(trails):
    iteration_accuracy = []
    for t in range(t_max):
        data = np.copy(data_random_classification)
        particles = generate_centroids(data, N_c, N_p)
        global_best = [[0]*N_c, 0]
        for particle in particles:
            local_best = [[0]*N_c, 0]
            fitness_scores = np.array([])
            for i, point in enumerate(data):
                distances = []
                for centroid in particle:
                    distances.append([euclidean_distance(point, centroid), centroid[-1]])
                point[-1] = distances[np.argmin(np.array(distances)[:, 0])][1]
                
            
            fitness_score = fitness(data, particle, N_c)
            
            local_best = [particle, fitness_score] if fitness_score > local_best[1] else local_best
        
            
        #update global best    
        global_best = local_best if local_best[1] > global_best[1] else global_best
        #update particle using pso update rule 
        
        for particle in particles:
            
            for centroid in particle:
                v = update_velocity(w, v, centroid[:-1], local_best[0][0][:-1], global_best[0][0][:-1])
                for n in range(N_c):
                    centroid[n] += v[n]

        right_classification = 0
        for point_guess, point_truth in zip(data, data_truth):
            if point_guess[Acc] == point_truth[Acc]:
                right_classification += 1

        accuracy = right_classification/len(data)
        iteration_accuracy.append(accuracy)
    accuracies.append(np.mean(iteration_accuracy))
average_accuracy = np.mean(accuracies)
print("Average accuracy: ", average_accuracy)

##Plots Exercise 3e
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(2, 1, 1)
# scatter1 = ax.scatter(data[:, 0],data[:, 1], c=data[:, -1])
# ax.set_title("PGO plot")
# ax.legend(handles=scatter1.legend_elements()[0], labels=['0', '1', '2'], title='classification')

# ax = fig.add_subplot(2, 1, 2)
# scatter2 = ax.scatter(data_truth[:, 0],data_truth[:, 1], c=data_truth[:, -1])
# ax.set_title("true labels")
# ax.legend(handles=scatter2.legend_elements()[0], labels=['0', '1', '2'], title='classification')

#plt.show()
