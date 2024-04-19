import numpy as np

"""
1_ Initialize the value function: We start by initializing the value function for all states.

2) Generate episodes: Similar to Monte Carlo methods, we generate episodes by following the given policy. However, in TD learning, we update the value function after each step, based on the observed immediate reward and the estimated value of the next state.

3) Update the value function: After each step in an episode, we update the value function for the current state using the observed immediate reward and the estimated value of the next state.

4) Ensure each state is visited often: To ensure that each state is visited often, we can use an epsilon-greedy exploration strategy. This means that with probability epsilon, we choose a random action, and with probability 1-epsilon, we choose the action according to the current policy. This ensures that the agent explores the environment sufficiently to estimate the values accurately.

5) Termination criteria: We can use a fixed number of episodes as the termination criteria. Alternatively, we can terminate when the change in the value function between consecutive iterations falls below a certain threshold.
"""

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
epsilon = 0.1  # Epsilon for epsilon-greedy exploration -> exploitation and exploration trade-off
threshold = 0.01

# Initialize value function
V = np.zeros_like(grid_world, dtype=float)


# Function to run one-step TD learning
def one_step_td(grid_world, policy, num_episodes, alpha, epsilon):
    for episode in range(num_episodes):
        state = (0, 0)  # Start state
        while True:
            # Choose action epsilon-greedily
            if np.random.rand() < epsilon:
                action = np.random.randint(4)  # Choose a random action
            else:
                action = policy[state]

            next_state, reward = take_action(state, action)
            # Update value function
            V[state[0], state[1]] += alpha * (
                reward + GAMMA * V[next_state[0], next_state[1]] - V[state[0], state[1]]
            )
            if grid_world[next_state[0], next_state[1]] in [
                -1,
                1,
                -2,
            ]:  # Terminal state
                break
            state = next_state

    return V


# Helper function to take action
def take_action(state, action):
    i, j = state
    if action == 0:  # Right
        next_state = (i, min(j + 1, grid_world.shape[1] - 1))
    elif action == 1:  # Left
        next_state = (i, max(j - 1, 0))
    elif action == 2:  # Up
        next_state = (max(i - 1, 0), j)
    else:  # Down
        next_state = (min(i + 1, grid_world.shape[0] - 1), j)

    reward = 0
    if grid_world[next_state[0], next_state[1]] == -1:  # Lightning bolt
        reward = -1
    elif grid_world[next_state[0], next_state[1]] == 1:  # Treasure chest
        reward = 1
    return next_state, reward


value_function = one_step_td(grid_world, policy, num_episodes, alpha, epsilon)

print("Value function at all states:")
print(value_function)
