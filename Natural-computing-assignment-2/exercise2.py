import numpy as np
import matplotlib.pyplot as plt

def f(x):
	return x**2

def update_velocity(w, v, x, x_hat, g_hat):
	return w * v + np.dot((a_1 * r_1) ,(x_hat - x)) + np.dot((a_2 * r_2), (g_hat - x))

def swarm_single(v, x, w, a_1, a_2, r_1, r_2, epochs):

    x_hat = x
    g_hat = x
    trajectory = []

    for i in range(epochs):
        #update velocity
        v = update_velocity(w, v, x, x_hat, g_hat)
        #update position
        x = x + v
        #update local best
        x_hat = x if f(x) < f(x_hat) else x_hat
        
        #update global best
        g_hat = x if f(x) < f(g_hat) else g_hat
        trajectory.append(x)

    return trajectory


# Setting 1
epochs = 30

v = 10
x = 20
w = 0.5
a_1, a_2 = 1.5, 1.5
r_1, r_2 = 0.5, 0.5

setting1 = swarm_single(v, x, w, a_1, a_2, r_1, r_2, epochs)

v = 10
x = 20
x_hat = x
g_hat = x
w = 0.7
a_1, a_2 = 1.5, 1.5
r_1, r_2 = 1.0, 1.0

setting2 = swarm_single(v, x, w, a_1, a_2, r_1, r_2, epochs)

plt.plot(setting1, setting1, marker="o", linestyle='--', label="Setting 1")
plt.plot(setting2, setting2, marker="x", linestyle="--", label="Setting 2")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()