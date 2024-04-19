import numpy as np
import matplotlib.pyplot as plt

from hw2_problem3_part1 import first_visit_mc
from hw2_problem3_part2 import one_step_td, take_action

grid_world = np.array(
    [
        [0, 0, 0, 0, 1],
        [0, -2, -1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, -2, -2, 0],
        [0, 0, 0, 0, 0],
    ]
)

policy = np.array(
    [
        [0, 0, 0, 0, 2],
        [1, 2, 1, 1, 2],
        [2, 2, 0, 0, 0],
        [2, 3, 3, 3, 2],
        [2, 0, 0, 2, 2],
    ]
)

num_episodes = 10000
GAMMA = 0.95
alpha = 0.1
epsilon = 0.1
threshold = 0.01


# Function to compute the true value function -> had to modify to comply with grid sizes
def compute_true_value_function(grid_world, policy, discount_factor):
    num_states = np.prod(grid_world.shape)
    num_actions = 4  # Number of actions: up, down, left, right

    # Define transition matrix T and reward matrix R
    T = np.zeros((num_states, num_states))
    R = np.zeros((num_states, num_states))

    # Fill in transition and reward matrices
    for i in range(grid_world.shape[0]):
        for j in range(grid_world.shape[1]):
            state = i * grid_world.shape[1] + j
            for action in range(num_actions):
                next_state, reward = take_action((i, j), action)
                next_state_index = next_state[0] * grid_world.shape[1] + next_state[1]
                T[
                    state, next_state_index
                ] += 0.85  # Probability of transitioning to the expected cell
                for k in range(num_actions):
                    if k != action:
                        next_state_k, reward_k = take_action((i, j), k)
                        next_state_index_k = (
                            next_state_k[0] * grid_world.shape[1] + next_state_k[1]
                        )
                        T[
                            state, next_state_index_k
                        ] += 0.05  # Probability of transitioning to one of the remaining three cells
                R[state, next_state_index] = reward

    # Add transitions from terminal states to the fictitious state
    terminal_states = np.where(np.logical_or(grid_world == -1, grid_world == 1))
    terminal_state_indices = [
        s[0] * grid_world.shape[1] + s[1] for s in zip(*terminal_states)
    ]
    fictitious_state_index = num_states - 1
    for state_index in terminal_state_indices:
        T[state_index, fictitious_state_index] = 1

    # Compute true value function using the analytical solution
    true_value_function = (
        np.linalg.inv(np.eye(num_states) - discount_factor * T)
        @ R
        @ np.ones((num_states, 1))
    )

    return true_value_function.reshape(grid_world.shape)


# Compute true value function (for comparison)
true_value_function = compute_true_value_function(grid_world, policy, GAMMA)

# Run Monte Carlo and TD learning algorithms
value_function_mc = first_visit_mc(grid_world, policy, num_episodes, threshold)
value_function_td = one_step_td(grid_world, policy, num_episodes, alpha, epsilon)

# Calculate errors
errors_mc = np.zeros(num_episodes)
errors_td = np.zeros(num_episodes)
for i in range(num_episodes):
    errors_mc[i] = np.linalg.norm(value_function_mc - true_value_function, ord="fro")
    errors_td[i] = np.linalg.norm(value_function_td - true_value_function, ord="fro")

# Plot sequence of errors
plt.plot(range(1, num_episodes + 1), errors_mc, label="Monte Carlo")
plt.plot(range(1, num_episodes + 1), errors_td, label="TD Learning")
plt.xlabel("Number of Episodes")
plt.ylabel("Error in Value Function (Frobenius Norm)")
plt.title("Sequence of Errors in Value Function")
plt.legend()
plt.grid(True)
plt.show()
