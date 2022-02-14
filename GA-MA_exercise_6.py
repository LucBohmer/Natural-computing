from scipy.spatial import distance
import random
import numpy as np
from matplotlib import pyplot as plt
from math import floor
import itertools

#cities = np.array([[1, 7], [2, 4], [4, 5], [3, 9], [4,6], [7,10], [1,6], [9,5], [8,2]])
cities1 = np.array(
    [
        [0.2554, 18.2366],
        [0.4339, 15.2476],
        [0.7377, 8.3137],
        [1.1354, 16.5638],
        [1.5820, 17.3030],
        [2.0913, 9.2924],
        [2.2631, 17.3392],
        [2.6373, 2.6425],
        [3.0040, 19.5712],
        [3.6684, 14.8018],
        [3.8630, 13.7008],
        [4.2065, 9.8224],
        [4.8353, 2.0944],
        [4.9785, 3.1596],
        [5.3754, 17.6381],
        [5.9425, 6.0360],
        [6.1451, 3.8132],
        [6.7782, 11.0125],
        [6.9223, 7.7819],
        [7.5691, 0.9378],
        [7.8190, 13.1697],
        [8.3332, 5.9161],
        [8.5872, 7.8303],
        [9.1224, 14.5889],
        [9.4076, 9.7166],
        [9.7208, 8.1154],
        [10.1662, 19.1705],
        [10.7387, 2.0090],
        [10.9354, 5.1813],
        [11.3707, 7.2406],
        [11.7418, 13.6874],
        [12.0526, 4.7186],
        [12.6385, 12.1000],
        [13.0950, 13.6956],
        [13.3533, 17.3524],
        [13.8794, 3.9479],
        [14.2674, 15.8651],
        [14.5520, 17.2489],
        [14.9737, 13.2245],
        [15.2841, 1.4455],
        [15.5761, 12.1270],
        [16.1313, 14.2029],
        [16.4388, 16.0084],
        [16.7821, 9.4334],
        [17.3928, 12.9692],
        [17.5139, 6.4828],
        [17.9487, 7.5563],
        [18.3958, 19.5112],
        [18.9696, 19.3565],
        [19.0928, 16.5453],
    ]
)

# Bayg 29
cities2 = np.array([[1150.0,1760.0],
[630.0,1660.0],
[40.0,2090.0],
[750.0,1100.0],
[750.0,2030.0],
[1030.0,2070.0],
[1650.0,650.0],
[1490.0,1630.0],
[790.0,2260.0],
[710.0,1310.0],
[840.0,550.0],
[1170.0,2300.0],
[970.0,1340.0],
[510.0,700.0],
[750.0,900.0],
[1280.0,1200.0],
[230.0,590.0],
[460.0,860.0],
[1040.0,950.0],
[590.0,1390.0],
[830.0,1770.0],
[490.0,500.0],
[1840.0,1240.0],
[1260.0,1500.0],
[1280.0,790.0],
[490.0,2130.0],
[1460.0,1420.0],
[1260.0,1910.0],
[360.0,1980.0]])
def fitness(cities, path):
    total_distance = 0
    for i in range(1, len(path)):
        x1= cities[path[i]][0]
        x2 = cities[path[i-1]][0] 
        y1= cities[path[i]][1] 
        y2 = cities[path[i-1]][1]
        total_distance += (((x2 - x1 )**2) + ((y2-y1)**2) )**0.5
    return 1 / total_distance


def generate_candidates(cities, n):
    permutations = [list(np.random.permutation([*range(len(cities))])) for i in range(n)]
    # list(itertools.permutations([*range(len(cities))]))
    # random.shuffle(permutations)
    return permutations


def mutation(child1, child2, p=0.05):
    if random.random() < p:
        i = random.sample(range(0, len(child1)), 4)
        child1[i[0]], child1[i[1]] = child1[i[1]], child1[i[0]]
        child2[i[2]], child2[i[3]] = child2[i[3]], child2[i[2]]

    return child1, child2


def crossover(parent1, parent2, cut, length):
    child1, child2 = np.empty(len(parent1), dtype=int), np.empty(
        len(parent1), dtype=int
    )
    child1.fill(-1)
    child2.fill(-1)

    child1[cut : cut + length] = parent1[cut : cut + length]
    child2[cut : cut + length] = parent2[cut : cut + length]

    middle_parent1 = parent1[cut : cut + length]
    not_in_middle_parent1 = [a for a in parent2 if a not in middle_parent1]

    middle_parent2 = parent2[cut : cut + length]
    not_in_middle_parent2 = [a for a in parent1 if a not in middle_parent2]

    j = 0
    for i in range(len(parent1)):
        if i < cut or i >= cut + length:
            child1[i] = not_in_middle_parent1[j]
            child2[i] = not_in_middle_parent2[j]
            j += 1
    return child1, child2

def two_opt_swap(route, i, k):
    new_route = route.copy()
    new_route[i:k] = route[i:k][::-1]
    return new_route

def local_search(route):
    improved = True
    while improved:
        improved = False
        best_fitness = fitness(cities, route)
        for i in range(len(cities)-1):
            for k in range(i+1, len(cities)-1):
                new_route = two_opt_swap(route, i, k)
                new_fitness = fitness(cities, new_route)
                if new_fitness > best_fitness:
                    route = new_route
                    best_fitness = new_fitness
                    improved = True
    return route

def genetic_algorithm(cities, epochs=1500, pop_size=10):
    best_fitness = []
    average_fitness = []
    # Initialize population
    candidates = generate_candidates(cities, pop_size)
    # Evaluate quality of each candidate
    # fitness_scores = []

    # for candidate in candidates:
    #     fitness_scores.append(fitness(cities, candidate))
    # Repeat until termination
    for epoch in range(epochs):
        # Select candidate solutions for reproduction
        print("Epoch:", epoch)
        children = []
        for i in range(0, pop_size, 2):
            tournament = random.sample([*range(len(candidates))], 2)
            if fitness(cities, candidates[tournament[0]]) >=  fitness(cities, candidates[tournament[1]]):
                parent1 = tournament[0]
            else:
                parent1 = tournament[1]

            tournament = random.sample([*range(len(candidates))], 2)
            if fitness(cities, candidates[tournament[0]]) >=  fitness(cities, candidates[tournament[1]]):
                parent2 = tournament[0]
            else:
                parent2 = tournament[1]


            # Crossover
            cut = random.randrange(1, floor(len(cities) / 2))
            length = random.randrange(2, floor(len(cities) / 2))
            
            child1, child2 = crossover(
                candidates[parent1], candidates[parent2], cut, length
            )

            # Mutation
            child1, child2 = mutation(child1, child2)

            children.append(child1)
            children.append(child2)

        # Candidate selection for next generation
        candidates = np.copy(children)

        fitness_scores = []
        distances = []
        for candidate in candidates:
            fitness_scores.append(fitness(cities, candidate))
        
        best_fitness.append(max(fitness_scores))
        average_fitness.append(sum(fitness_scores)/len(fitness_scores))

    return candidates[np.array(fitness_scores).argsort()[-1:][::-1][0]], best_fitness, average_fitness


def memetic_algorithm(cities, epochs=1500, pop_size=10):
    best_fitness = []
    average_fitness = []
    # Initialize population
    initial_candidates = generate_candidates(cities, pop_size)
    candidates = []
    for individual in initial_candidates:
        candidates.append(local_search(individual))
    # Evaluate quality of each candidate

    # for candidate in candidates:
    #     fitness_scores.append(fitness(cities, candidate))
    # Repeat until termination
    for epoch in range(epochs):
        print("Epoch:", epoch)
        # Select candidate solutions for reproduction
        #best_index = np.array(fitness_scores).argsort()[-pop_size:][::-1]

        children = []
        # cut = 13
        # length = 24
        for i in range(0, pop_size, 2):
            tournament = random.sample([*range(len(candidates))], 2)
            if fitness(cities, candidates[tournament[0]]) >=  fitness(cities, candidates[tournament[1]]):
                parent1 = tournament[0]
            else:
                parent1 = tournament[1]

            tournament = random.sample([*range(len(candidates))], 2)
            if fitness(cities, candidates[tournament[0]]) >=  fitness(cities, candidates[tournament[1]]):
                parent2 = tournament[0]
            else:
                parent2 = tournament[1]


            # Crossover
            cut = random.randrange(1, floor(len(cities) / 2))
            length = random.randrange(2, floor(len(cities) / 2))
            
            child1, child2 = crossover(
                candidates[parent1], candidates[parent2], cut, length
            )

            # Mutation
            child1, child2 = mutation(child1, child2)

            child1 = local_search(child1)
            child2 = local_search(child2)

            children.append(child1)
            children.append(child2)

        # Candidate selection for next generation
        candidates = np.copy(children)

        fitness_scores = []
        distances = []
        for candidate in candidates:
            fitness_scores.append(fitness(cities, candidate))
        
        best_fitness.append(max(fitness_scores))
        average_fitness.append(sum(fitness_scores)/len(fitness_scores))
    return candidates[np.array(fitness_scores).argsort()[-1:][::-1][0]], best_fitness, average_fitness




cities = cities2
best_fitnesses = []
average_fitnesses = []
# ---- Genetic
for i in range(10):
    best, best_fitness, average_fitness = genetic_algorithm(cities)
    best_fitnesses.append(best_fitness)
    average_fitnesses.append(average_fitness)
for j in range(len(best_fitnesses)):
    plt.plot(best_fitnesses[j], label=str(j))

plt.title("Memetic Algorithm - Best fitness")
plt.xlabel("Iterations")
plt.ylabel("Fitness")
plt.show()
for j in range(len(average_fitnesses)):
    plt.plot(average_fitnesses[j], label=str(j))

plt.title("Memetic Algorithm - Average fitness")
plt.xlabel("Iterations")
plt.ylabel("Fitness")
plt.show()
for i in range(1, len(best)):
    plt.plot(
        [cities[best[i - 1]][0], cities[best[i]][0]],
        [cities[best[i - 1]][1], cities[best[i]][1]],
        "r-",
    )
for i in range(len(best)):
    plt.annotate(str(i), (cities[i][0], cities[i][1]))
plt.show()

# ---- Memetic
for i in range(10):
    best, best_fitness, average_fitness = memetic_algorithm(cities)
    best_fitnesses.append(best_fitness)
    average_fitnesses.append(average_fitness)
for j in range(len(best_fitnesses)):
    plt.plot(best_fitnesses[j], label=str(j))

plt.title("Memetic Algorithm - Best fitness")
plt.xlabel("Iterations")
plt.ylabel("Fitness")
plt.show()
for j in range(len(average_fitnesses)):
    plt.plot(average_fitnesses[j], label=str(j))

plt.title("Memetic Algorithm - Average fitness")
plt.xlabel("Iterations")
plt.ylabel("Fitness")
plt.show()
for i in range(1, len(best)):
    plt.plot(
        [cities[best[i - 1]][0], cities[best[i]][0]],
        [cities[best[i - 1]][1], cities[best[i]][1]],
        "r-",
    )
for i in range(len(best)):
    plt.annotate(str(i), (cities[i][0], cities[i][1]))
plt.show()


