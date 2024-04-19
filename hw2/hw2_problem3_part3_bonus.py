import numpy as np

"""
To compute the true value function analytically, we can view the problem as an infinite-horizon one and apply the analytical solution for policy evaluation
We can introduce a new state that acts as a placeholder for transitions from terminal states. 
Any action taken in the terminal states will transition to this fake state. All actions in this new state will keep the agent in the same state and have a reward of 0.
"""

# ! HAS TO BE SQUARE GRIDS FOR THIS TO WORK

grid = np.array(
    [
        [0, 0, 0, 0, 1, 0],  # Random fake state represented by 0th state row
        [0, 0, 0, 0, 0, 0],
        [0, 0, -2, -1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, -2, -2, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]
)

policy = np.array(
    [
        [0, 0, 0, 0, 2, 0],  # Random fake state represented by 0th action row
        [1, 2, 1, 1, 2, 0],
        [2, 2, 0, 0, 0, 0],
        [2, 3, 3, 3, 2, 0],
        [2, 0, 0, 2, 2, 0],
        [0, 0, 0, 0, 0, 0],
    ]
)

discount_factor = 0.95


# Function to compute the true value function
def true_value_function(grid, discount_factor):
    num_states = np.prod(grid.shape)
    num_actions = 4

    # Define transition matrix and reward matrix T and R respectively (T[s, s'] and R[s, s'])
    T = np.zeros((num_states, num_states))
    R = np.zeros((num_states, num_states))

    # Populate T and R
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            state = i * grid.shape[1] + j
            for action in range(num_actions):
                next_state, reward = take_action((i, j), action)
                next_state_index = next_state[0] * grid.shape[1] + next_state[1]
                T[
                    state, next_state_index
                ] += 0.85  # Probability of transitioning to the expected cell
                for k in range(num_actions):
                    if k != action:
                        next_state_k, _ = take_action((i, j), k)
                        next_state_index_k = (
                            next_state_k[0] * grid.shape[1] + next_state_k[1]
                        )
                        T[
                            state, next_state_index_k
                        ] += 0.05  # Probability of transitioning to one of the remaining three cells
                R[state, next_state_index] = reward

    # Add transitions from terminal states to the fake state
    terminal_states = np.where(np.logical_or(grid == -1, grid == 1))
    terminal_state_indices = [
        s[0] * grid.shape[1] + s[1] for s in zip(*terminal_states)
    ]
    fake_state_index = num_states - 1
    for state_index in terminal_state_indices:
        T[state_index, fake_state_index] = 1

    # Compute true value function using the analytical solution
    true_value_function = (
        np.linalg.inv(np.eye(num_states) - discount_factor * T)
        @ R
        @ np.ones((num_states, 1))
    )

    return true_value_function.reshape(grid.shape)


# Helper function to take action
def take_action(state, action):
    i, j = state
    if action == 0:  # Right
        next_state = (i, min(j + 1, grid.shape[1] - 1))
    elif action == 1:  # Left
        next_state = (i, max(j - 1, 0))
    elif action == 2:  # Up
        next_state = (max(i - 1, 0), j)
    else:  # Down
        next_state = (min(i + 1, grid.shape[0] - 1), j)

    reward = 0
    if grid[next_state[0], next_state[1]] == -1:  # Lightning bolt
        reward = -1
    elif grid[next_state[0], next_state[1]] == 1:  # Treasure chest
        reward = 1
    return next_state, reward


tvf = true_value_function(grid, discount_factor)

print("True value function at all states:")
print(tvf)
