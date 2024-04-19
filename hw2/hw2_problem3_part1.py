import numpy as np

"""
1) Generate episodes: We start by generating episodes by following the given policy. During each episode, we record the states, actions, and rewards encountered.

2) Calculate returns: Once an episode terminates, we calculate the return (total discounted reward) for each state visited in the episode.

3) Update state values: Using the returns obtained, we update the value function for each visited state using the first-visit Monte Carlo update rule.

4) Ensure each state is visited often: To ensure that each state is visited often, we can either set a fixed number of episodes to run or terminate the algorithm when the change in value function between consecutive iterations falls below a certain threshold.

5) Termination criteria: For this task, we can use a threshold on the change in value function between consecutive iterations as the termination criteria. Once the change in value function becomes smaller than a predefined threshold, we consider the algorithm to have converged.
"""

grid = np.array(
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
threshold = 0.01

# Initialize value function
V = np.zeros_like(  # zeros_like because zeros creates integers and we need floats
    grid, dtype=float
)
returns = {(i, j): [] for i in range(grid.shape[0]) for j in range(grid.shape[1])}
state_visits = {(i, j): 0 for i in range(grid.shape[0]) for j in range(grid.shape[1])}


# Function to run Monte Carlo
def first_visit_mc(grid, policy, num_episodes, threshold):
    prev_V = np.copy(V)
    for episode in range(num_episodes):
        # Generate an episode
        states = []
        actions = []
        rewards = []
        state = (0, 0)  # Start state
        while True:
            states.append(state)
            action = policy[state]
            actions.append(action)
            next_state, reward = take_action(state, action)
            rewards.append(reward)
            if grid[next_state[0], next_state[1]] in [-1, 1, -2]:  # Terminal state
                break
            state = next_state

        # Calculate returns and update value function
        G = 0
        for t in range(len(states) - 1, -1, -1):  # Iterate over states in reverse order
            state = states[t]
            G = GAMMA * G + rewards[t]
            if state not in states[:t]:  # First visit to the state
                returns[state].append(G)
                state_visits[state] += 1
                V[state[0], state[1]] = np.mean(returns[state])

        # Check termination criteria
        if episode > 0:
            delta = np.max(np.abs(V - prev_V))
            if delta < threshold:
                break
        prev_V = np.copy(V)

    return V


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


# Run first-visit Monte Carlo algorithm
value_function = first_visit_mc(grid, policy, num_episodes, threshold)

print("Value function at all states:")
print(value_function)
