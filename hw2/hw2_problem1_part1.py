import numpy as np

# Define the grid
grid = [  # this is the state space
    [0, 0, 0, 0, 1],
    [0, -2, -1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, -2, -2, 0],
    [0, 0, 0, 0, 0],
]

actions = ["up", "down", "left", "right"]

# Discount factor
GAMMA = 0.95


# Define the transition function/probabilities
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
        if grid[row][col] == -1:  # if the state is lightning then return -1
            return -1
        elif grid[row][col] == 1:  # if the state is the goal then return 1
            return 1
    return 0  # otherwise return 0


# Value iteration
def value_iteration(theta):
    value = np.zeros((len(grid), len(grid[0])))  # initialize value function to 0
    delta = float("inf")
    while delta > theta:  # break when the change in value function is less than theta
        delta = 0
        for row in range(len(grid)):
            for col in range(len(grid[0])):
                state = (row, col)
                old_value = value[row][col]
                if grid[row][col] != -2:  # Exclude unreachable states
                    value[row][col] = max(
                        [
                            sum(
                                [
                                    transition_prob(state, action)[next_state]
                                    * (
                                        reward(next_state)
                                        + GAMMA * value[next_row][next_col]
                                    )
                                    for next_state in transition_prob(state, action)
                                    for next_row, next_col in [next_state]
                                    if 0 <= next_row < len(grid)
                                    and 0 <= next_col < len(grid[0])
                                ]
                            )
                            for action in actions
                        ]
                    )
                    delta = max(delta, abs(value[row][col] - old_value))
    return value


optimal_value = value_iteration(0.01)

print("Optimal Value Function:")
for row in range(len(grid)):
    for col in range(len(grid[0])):
        state = (row, col)
        value = optimal_value[row][col]
        if grid[row][col] == -2:  # Unreachable states (mountains and lightning) -> -2
            print("X ", end="")
        else:
            print(f"{value:.2f} ", end="")
    print()

# Compute the optimal policy
policy = np.zeros((len(grid), len(grid[0])), dtype=object)
for row in range(len(grid)):
    for col in range(len(grid[0])):
        if grid[row][col] != -2:  # Exclude unreachable states
            state = (row, col)
            policy[row][col] = max(
                actions,
                key=lambda action: sum(
                    [
                        transition_prob(state, action)[next_state]
                        * (
                            reward(next_state)
                            + GAMMA * optimal_value[next_row][next_col]
                        )
                        for next_state in transition_prob(state, action)
                        for next_row, next_col in [next_state]
                        if 0 <= next_row < len(grid) and 0 <= next_col < len(grid[0])
                    ]
                ),
            )

print("\nOptimal Policy:")
for row in range(len(grid)):
    for col in range(len(grid[0])):
        if grid[row][col] == -2:
            print("X ", end="")
        else:
            print(f"{policy[row][col]} ", end="")
    print()
