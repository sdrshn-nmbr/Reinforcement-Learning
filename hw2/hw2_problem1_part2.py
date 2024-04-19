import numpy as np

# Define the grid
grid = [
    [0, 0, 0, 0, 1],
    [0, -2, -1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, -2, -2, 0],
    [0, 0, 0, 0, 0],
]

actions = ["up", "down", "left", "right"]

# Discount factor
GAMMA = 0.95


# Define the transition probabilities
def transition_prob(state, action):
    row, col = state
    prob = {}
    if action == "up":
        if row == 0:
            prob[(row, col)] = 1.0
        else:
            prob[(row - 1, col)] = 0.85
            prob[(row, col - 1)] = 0.05
            prob[(row, col + 1)] = 0.05
            prob[(row, col)] = 0.05
    elif action == "down":
        if row == len(grid) - 1:
            prob[(row, col)] = 1.0
        else:
            prob[(row + 1, col)] = 0.85
            prob[(row, col - 1)] = 0.05
            prob[(row, col + 1)] = 0.05
            prob[(row, col)] = 0.05
    elif action == "left":
        if col == 0:
            prob[(row, col)] = 1.0
        else:
            prob[(row, col - 1)] = 0.85
            prob[(row - 1, col)] = 0.05
            prob[(row + 1, col)] = 0.05
            prob[(row, col)] = 0.05
    elif action == "right":
        if col == len(grid[0]) - 1:
            prob[(row, col)] = 1.0
        else:
            prob[(row, col + 1)] = 0.85
            prob[(row - 1, col)] = 0.05
            prob[(row + 1, col)] = 0.05
            prob[(row, col)] = 0.05
    return prob


# Define the reward function
def reward(state):
    row, col = state
    if 0 <= row < len(grid) and 0 <= col < len(grid[0]):
        if grid[row][col] == -1:
            return -1
        elif grid[row][col] == 1:
            return 1
    return 0


# Policy iteration algorithm
def policy_iteration():
    policy = np.random.choice(
        actions, size=(len(grid), len(grid[0]))
    )  # Initialize policy randomly
    while True:
        value = policy_eval(policy, 0.01)  # Policy evaluation step
        new_policy = policy_improvement(value)  # Policy improvement step
        if np.array_equal(new_policy, policy):  # Check for convergence
            break
        policy = new_policy
    return value, policy


# Policy evaluation step
def policy_eval(policy, theta):
    value = np.zeros((len(grid), len(grid[0])))
    while True:
        delta = 0
        for row in range(len(grid)):
            for col in range(len(grid[0])):
                state = (row, col)
                old_value = value[row][col]
                if grid[row][col] != -2:
                    action = policy[row][col]
                    value[row][col] = sum(
                        [
                            transition_prob(state, action)[next_state]
                            * (reward(next_state) + GAMMA * value[next_row][next_col])
                            for next_state in transition_prob(state, action)
                            for next_row, next_col in [next_state]
                            if 0 <= next_row < len(grid)
                            and 0 <= next_col < len(grid[0])
                        ]
                    )
                    delta = max(delta, abs(value[row][col] - old_value))
        if delta < theta:
            break
    return value


# Policy improvement step
def policy_improvement(value):
    new_policy = np.zeros((len(grid), len(grid[0])), dtype=object)
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            if grid[row][col] != -2:
                state = (row, col)
                new_policy[row][col] = max(
                    actions,
                    key=lambda action: sum(
                        [
                            transition_prob(state, action)[next_state]
                            * (reward(next_state) + GAMMA * value[next_row][next_col])
                            for next_state in transition_prob(state, action)
                            for next_row, next_col in [next_state]
                            if 0 <= next_row < len(grid)
                            and 0 <= next_col < len(grid[0])
                        ]
                    ),
                )
    return new_policy


# Compute optimal value function and policy
optimal_value_policy, optimal_policy_policy = policy_iteration()

print("Optimal Value Function (Policy Iteration):")
for row in range(len(grid)):
    for col in range(len(grid[0])):
        state = (row, col)
        value = optimal_value_policy[row][col]
        if grid[row][col] == -2:  # Unreachable states (mountains and lightning) -> -2
            print("X ", end="")
        else:
            print(f"{value:.2f} ", end="")
    print()

print("\nOptimal Policy (Policy Iteration):")
for row in range(len(grid)):
    for col in range(len(grid[0])):
        if grid[row][col] == -2:
            print("X ", end="")
        else:
            print(f"{optimal_policy_policy[row][col]} ", end="")
    print()
