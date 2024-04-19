import numpy as np
import matplotlib.pyplot as plt
from hw1_problem4_part5 import compute_stationary_distribution

transition_matrix = np.array([
    [0.6, 0.2, 0.2],
    [0.1, 0, 0.9],
    [0.3, 0, 0.7]
])

# Compute the stationary distribution
stationary_distribution = compute_stationary_distribution(transition_matrix)

# Initialize the initial state distribution
initial_state_distribution = np.array([1, 0, 0])

# Define the number of time steps -> from t=0 to t=100
num_time_steps = 101

# Initialize an array to store the differences over time
diffs_over_time = np.zeros(num_time_steps)

# Perform consecutive updates and calculate differences
curr_state_distribution = initial_state_distribution.copy()

for _ in range(num_time_steps):
    diffs_over_time[_] = np.linalg.norm(curr_state_distribution - stationary_distribution, ord=1)
    curr_state_distribution = np.dot(curr_state_distribution, transition_matrix)

# Plot
plt.plot(range(num_time_steps), diffs_over_time)
plt.xlabel('Time Steps')
plt.ylabel('L1 Norm Difference')
plt.title('Convergence to Stationary Distribution')
plt.grid(True)
plt.show()